import os
import time
import random
import logging

from enum import Enum
from pathlib import Path
from inspect import signature
from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional

import numpy
import torch
import torch_geometric
from torch import nn
from tqdm import tqdm

from dhg.nn import GCNConv, HGNNPConv
from torch.nn import TransformerEncoderLayer
from torch_geometric.nn import SAGEConv, GraphSAGE, GCN, GAT

from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, SCIP_PARAMSETTING

_DEVICE: Optional[torch.device] = None
_SEED = None


class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.WARNING: "\033[31m",  # RED
        logging.ERROR: "\033[31;1m",  # HIGHRED
        logging.INFO: "\033[32m",  # GREEN
        logging.DEBUG: "\033[34m",  # BLUE
    }
    RESET_CODE = "\033[0m"

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET_CODE}"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = ColorFormatter(fmt="[%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        raise RuntimeError("Device not initialized. Call init() first.")
    return _DEVICE


def get_current_seed() -> int:
    global _SEED
    if _SEED is None:
        raise RuntimeError("Seed not initialized. Call init() first.")
    return _SEED


def init(
    device: Optional[torch.device] = None,
    cuda_index: int = 0,
    verbose: bool = True,
    reproducibility: bool = False,
    seed=42,
):
    os.environ["PYTHONHASHSEED"] = str(seed)

    global _DEVICE, _SEED
    _SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device is not None:
        _DEVICE = device
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > cuda_index:
            if reproducibility:
                logger.warning(
                    "You have enabled the reproducibility feature, which uses a deterministic non-optimized algorithm, greatly affecting the running efficiency"
                )
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.use_deterministic_algorithms(True)
                torch.set_deterministic_debug_mode("warn")
            _DEVICE = torch.device(f"cuda:{cuda_index}")
            torch.cuda.set_device(cuda_index)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            numpy.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch_geometric.seed_everything(seed)

            if verbose:
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(cuda_index)} (Index: {cuda_index})")
        else:
            _DEVICE = torch.device("cpu")
            if verbose:
                logger.warning("CUDA device not available. Using CPU.")


class BaseTabuCol(ABC):
    def __init__(self, edges, k, tabu_tenure=30, max_iter=1000):
        self.edges = edges
        self.n = self._get_vertex_count()
        self.k = k
        self.tabu_tenure = tabu_tenure
        self.max_iter = max_iter

        # For hypergraphs: map vertices to their hyperedges
        self.vertex_to_edges = [[] for _ in range(self.n)]
        for idx, edge in enumerate(self.edges):
            for v in edge:
                self.vertex_to_edges[v].append(idx)

    def _get_vertex_count(self):
        all_vertices = set()
        for edge in self.edges:
            all_vertices.update(edge)
        return max(all_vertices) + 1 if all_vertices else 0

    def initial_solution(self):
        return [random.randint(1, self.k) for _ in range(self.n)]

    @abstractmethod
    def evaluate(self, solution):
        pass

    @abstractmethod
    def calculate_delta(self, u, old_color, new_color, solution):
        pass

    def solve(self, time_limit=None):
        start_time = time.time()
        current_sol = self.initial_solution()
        best_sol = current_sol.copy()
        best_conflicts = self.evaluate(current_sol)

        tabu_list = {}
        for iter in range(self.max_iter):
            best_move = None
            best_delta = float("inf")

            # Explore neighborhood
            for u in range(self.n):
                original_color = current_sol[u]
                for new_color in range(1, self.k + 1):
                    if new_color == original_color:
                        continue

                    delta = self.calculate_delta(u, original_color, new_color, current_sol)

                    # Tabu check
                    if (u in tabu_list) and (new_color == tabu_list[u][0]) and (iter < tabu_list[u][1]):
                        continue

                    # Update best move
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (u, new_color)

            # Apply move
            if best_move:
                u, new_color = best_move
                old_color = current_sol[u]
                current_sol[u] = new_color
                tabu_list[u] = (old_color, iter + self.tabu_tenure)

                # Update best solution
                new_conflicts = best_conflicts + best_delta
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_sol = current_sol.copy()
                    logger.debug(f"Iter {iter+1:3d}: Conflicts reduced to {best_conflicts}")
                    if best_conflicts == 0:
                        break

            logger.debug(f"Iter {iter+1:3d}: conflicts: {best_conflicts}")
            
            if time_limit is not None:
                elapsed = time.time() - start_time
                if elapsed >= time_limit:
                    logger.info("⌛ Time limit exceeded after solution discovery")
                    break

        return best_sol, best_conflicts


class BaseSCIPSolver(ABC):
    def __init__(self, edge_list, pre_solve=True):
        self.edge_list = edge_list
        self.V = list(sorted({v for e in edge_list for v in e}))
        self.model = Model("Problem")
        self.solution_history = []
        self.start_time = None
        self.pre_solve = pre_solve

    @abstractmethod
    def _add_variables(self):
        pass

    @abstractmethod
    def _add_constraints(self):
        pass

    @abstractmethod
    def _set_objective(self):
        pass

    @abstractmethod
    def _get_solution_metrics(self, sol):
        pass

    @abstractmethod
    def _extract_solution(self, sol):
        pass

    def _setup_model(self, time_limit, sol_limit):
        self.model.setRealParam("limits/time", time_limit)
        logger.debug(f"set scip time limit: {time_limit}")
        if sol_limit is not None:
            self.model.setRealParam("limits/primal", sol_limit)
            logger.debug(f"set scip solution limit: {sol_limit}")
        self.model.setPresolve(SCIP_PARAMSETTING.OFF) if not self.pre_solve else None

    class SolutionTracker(Eventhdlr):
        def __init__(self, solver):
            super().__init__()
            self.solver = solver

        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexec(self, event):
            elapsed_time = time.time() - self.solver.start_time
            current_sol = self.model.getBestSol()
            metrics = self.solver._get_solution_metrics(current_sol)
            self.solver.solution_history.append((elapsed_time, metrics))
            print(f"[{elapsed_time:.2f}s] Current solution: {metrics}")

    def solve(self, time_limit=3600, sol_limit=None):
        self.start_time = time.time()
        self.solution_history = []
        self._setup_model(time_limit, sol_limit)
        self._add_variables()
        self._add_constraints()
        self._set_objective()

        tracker = self.SolutionTracker(self)
        self.model.includeEventhdlr(tracker, "SolutionTracker", "Tracks solutions")

        self.model.optimize()

        if self.model.getStatus() == "optimal":
            sol = self.model.getBestSol()
            final_time = time.time() - self.start_time
            metrics = self._get_solution_metrics(sol)
            print(f"\n[{final_time:.2f}s] Optimal solution found. Metrics: {metrics}")
            return self._extract_solution(sol)
        else:
            return self.model.getBestSol()


class Datasets(Enum):
    Graph_Cora = "cora_graph"
    Graph_Citeseer = "citeseer_graph"
    Graph_Amazon_PC = "amazon_electronics_computers_graph"
    Graph_Amazon_Photo = "amazon_electronics_photo_graph"
    Graph_Pubmed = "pubmed_graph"
    Graph_dblp = "dblp_graph"
    Graph_bat = "bat_graph"
    Graph_eat = "eat_graph"
    Graph_uat = "uat_graph"

    Hypergraph_Cora = "cora_hypergraph"
    Hypergraph_high = "high_hypergraph"
    Hypergraph_primary = "primary_hypergraph"
    Hypergraph_pubmed = "pubmed_hypergraph"
    Hypergraph_cooking200 = "cooking_200_hypergraph"

    @property
    def path(self):
        current_path = Path(__file__).parent
        type_path = self.value.split("_")[-1]
        data_path = os.path.join(current_path, "..", "data", type_path, f"{self.value}.txt")
        return data_path
    
    @property
    def type(self):
        return self.value.split("_")[-1]


class LayerType(Enum):
    TRANSFORMERENCODER = 0
    GCNCONV = 1
    HGNNPCONV = 2
    LINEAR = 3
    SAGECONV = 4
    GRAPHSAGE = 5
    GCN = 6
    GAT = 7


class Layer(nn.Module):
    _PARAM_MAPPING = {
        LayerType.GCNCONV: {"drop_rate": "drop_rate"},
        LayerType.HGNNPCONV: {"drop_rate": "drop_rate"},
        LayerType.GRAPHSAGE: {"drop_rate": "dropout"},
    }

    def __init__(self, layer_type: LayerType, in_channels: int, out_channels: int, act: Callable = nn.ReLU(), **kwargs):
        super().__init__()
        self.layer_type = layer_type

        params = self._adapt_parameters(kwargs)

        if layer_type == LayerType.GCNCONV:
            self.layer = GCNConv(
                in_channels,
                out_channels,
                use_bn=params.get("use_bn", True),
                drop_rate=params.get("dropout", 0.0),
                is_last=params.get("last_conv", False),
            )
        elif layer_type == LayerType.HGNNPCONV:
            self.layer = HGNNPConv(
                in_channels,
                out_channels,
                use_bn=params.get("use_bn", True),
                drop_rate=params.get("dropout", 0.0),
                is_last=params.get("last_conv", False),
            )
        elif layer_type == LayerType.SAGECONV:
            self.layer = SAGEConv(in_channels=in_channels, out_channels=out_channels, **self._filter_params(SAGEConv, params))
        elif layer_type == LayerType.GRAPHSAGE:
            self.layer = GraphSAGE(in_channels=in_channels, out_channels=out_channels, **self._filter_params(GraphSAGE, params))
        elif layer_type == LayerType.GCN:
            self.layer = GCN(in_channels=in_channels, out_channels=out_channels, **self._filter_params(GCN, params))
        elif layer_type == LayerType.GAT:
            self.layer = GAT(in_channels=in_channels, out_channels=out_channels, **self._filter_params(GCN, params))
        elif layer_type == LayerType.TRANSFORMERENCODER:
            if in_channels != out_channels:
                raise ValueError("TransformerEncoderLayer need the same dim between in and out channels!")
            self.layer = TransformerEncoderLayer(d_model=in_channels, **self._filter_params(TransformerEncoderLayer, params))
        elif layer_type == LayerType.LINEAR:
            self.linear = nn.Linear(in_channels, out_channels)
            self.bn = nn.BatchNorm1d(in_channels) if params.get("use_bn", True) else None
            self.dropout = nn.Dropout(params.get("dropout", 0.0)) if params.get("dropout", 0.0) > 0.0 else None
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        self.activation = act

    def _adapt_parameters(self, params: dict) -> dict:
        mapping = self._PARAM_MAPPING.get(self.layer_type, {})
        return {mapping.get(k, k): v for k, v in params.items()}

    def _filter_params(self, cls, params: dict) -> dict:
        sig = signature(cls.__init__)
        valid_params = sig.parameters.keys()
        return {k: v for k, v in params.items() if k in valid_params}

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, edge_weight=None, **kwargs) -> torch.Tensor:
        if self.layer_type in [LayerType.GCNCONV, LayerType.HGNNPCONV]:
            x = self.layer(x, graph)
        elif self.layer_type in [LayerType.SAGECONV, LayerType.GRAPHSAGE, LayerType.GCN, LayerType.GAT]:
            x = self.layer(x, edge_index, edge_weight=edge_weight)
        elif self.layer_type in [LayerType.TRANSFORMERENCODER]:
            x = self.layer(x)
        elif self.layer_type == LayerType.LINEAR:
            if self.bn is not None:
                x = self.bn(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.linear(x)

        return x


def train(net, X, graph, optimizer, loss_fn, clip_grad=False, **kwargs):
    """Conduct a round of neural network training
    
    Args:
        kwargs: 
            `loss_fn`'s kwargs
    
    """
    net.train()
    optimizer.zero_grad()
    outs = net.forward(X, graph, **kwargs)
    loss:torch.Tensor = loss_fn(outs=outs, **kwargs)
    loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad.clip_grad_norm_(net.parameters(), max_norm=5.5)
    optimizer.step()
    return loss.detach().item(), outs


def run(net, X, graph, num_epochs, loss_fn, lr, opt, **kwargs):
    """Starting the training loop
    Args:
        kwargs: 
            - clip_grad (bool): Whether to clip the gradient or not
            - see `src/core/train`
    """
    optimizer = (
        torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-8)
        if opt == "AdamW"
        else torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-8)
    )
    for epoch in tqdm(range(1, num_epochs + 1)):
        kwargs = {**kwargs, "epoch": epoch}
        loss, outs = train(net, X, graph, optimizer, loss_fn, num_epochs=num_epochs, **kwargs)
    del net, optimizer
    
    return loss, outs


# modified
def run_qubo(type: Literal["coloring", "partitioning", "maxcut", "mis"], net, X, graph, num_epochs, loss_fn, lr, opt: Literal["Adam", "AdamW"] = "AdamW", evaluate=False, **kwargs):
    """Solve graph coloring and graph partitioning problems
    
    Args:
        evaluate (bool):
            Whether to return the detailed evaluation results of the corresponding task
        kwargs: 
            - see `src/core/run`
    TODO:
        [@2024/4/19]
            Seems that the `loss_fn` parameter is not necessary, considered for deletion
        [@2024/4/30]
            Change this parameter to optional
    """

    if type == "coloring":
        from .coloring.utils import coloring_construct_Q as construct_Q
        from .coloring.utils import coloring_evaluate as evaluate_f
        from .coloring.loss_gini import loss_coloring_gini_qubo as _loss_fn
    elif type == "partitioning":
        from .partitioning.utils import partitioning_construct_Q as construct_Q
        from .partitioning.utils import partitioning_evaluate as evaluate_f
        from .partitioning.loss_gini import loss_partitioning_gini_qubo as _loss_fn
    elif type == "max_cut":
        from .max_cut.utils import maxcut_construct_Q as construct_Q
        from .max_cut.utils import maxcut_evaluate as evaluate_f
        from .max_cut.loss_gini import loss_maxcut_gini_qubo as _loss_fn
    elif type == "mis":
        from .mis.utils import mis_construct_Q as construct_Q
        from .mis.utils import mis_evaluate as evaluate_f
        from .mis.loss_gini import loss_mis_gini_qubo as _loss_fn
    else:
        raise ValueError("type parameter Error")
    
    Q = construct_Q(graph)
    edge_index = torch.tensor(graph.e[0], dtype=torch.long, device=X.device).t().contiguous()
    lf = loss_fn if loss_fn is not None else _loss_fn
    if type == "mds":
        B = construct_B(graph)
        loss, outs = run(
            net, X, graph, num_epochs, lf, lr, opt,
            edge_index=edge_index, Q=Q, B=B, **kwargs
        )
    else:
        loss, outs = run(
            net, X, graph, num_epochs, lf, lr, opt,
            edge_index=edge_index, Q=Q, **kwargs
        )

    if type == "coloring":
        evaluate_results = evaluate_f(outs[0], graph)
    else:
        evaluate_results = evaluate_f(outs, graph)

    if evaluate:
        return loss, outs, evaluate_results
    
    return loss, outs


# modified
def run_graph_pubo(
    type: Literal["coloring", "partitioning", "maxcut", "mis", "mds"], net, X, graph, num_epochs, loss_fn, lr, opt: Literal["Adam", "AdamW"] = "adamW", evaluate=False, simple=False, **kwargs
):
    """Solve graph PUBO problems

    Args:
        evaluate (bool):
            Whether to return the detailed evaluation results of the corresponding task
        simple (bool):
           Modified according to the method of generating graphs based on hypergraphs, see `Note` and [DHG](https://deephypergraph.readthedocs.io/en/latest/api/dhg.html?highlight=hypergcn#graph)
        kwargs: see `src/core/run`
    """
    if type == "mds":
        from .mds_pubo.utils import mds_evaluate as evaluate
        from .mds_pubo.loss_gini import loss_mds_gini_pubo as loss_fn
    else:
        raise ValueError("type parameter Error")
    
    edges = graph.e[0]
    nodes = sorted(graph.v)
    node2idx = {v: i for i, v in enumerate(nodes)}

    adjacency = [set([i]) for i in range(len(nodes))]
    for u, v in edges:
        iu = node2idx[u]
        iv = node2idx[v]    
        adjacency[iu].add(iv)
        adjacency[iv].add(iu)

    closed_nbh = [
        torch.tensor(sorted(list(nbh)), dtype=torch.long, device=X.device)
        for nbh in adjacency
    ]

    edge_index = torch.tensor(graph.e[0], dtype=torch.long).t().contiguous()

    loss, outs = run(net, X, graph, num_epochs, loss_fn, lr, opt, edge_index=edge_index, closed_nbh=closed_nbh, **kwargs)
    evalute_results = evaluate(outs, graph)

    if evaluate:
        return loss, outs, evalute_results
    
    return loss, outs, None