from collections import defaultdict
import dhg
import time
import torch
import logging

from tqdm import tqdm
from typing import Dict, Literal, Optional, Tuple
from ..core import BaseTabuCol, BaseSCIPSolver
from pyscipopt import Model, quicksum, Eventhdlr, SCIP_EVENTTYPE

logger = logging.getLogger(__name__)


def coloring_construct_Q(graph, full=False):
    """Constructs graph coloring as an `OH-QUBO` formulation requiring matrix Q.

    The coloring problem is modeled as `Reduce_sum(X^T·Q⊙X)`, where:
    - `Reduce_sum` performs element-wise matrix summation.
    - X represents the solution vector/matrix.
    - Q is the problem-specific design matrix.
        + In undirected graph coloring problem, the Q is the upper triangular matrix (excluding the diagonal) of the adjacency matrix.
    """
    A = graph.A.to_dense().float()
    if not full:
        Q = A.triu(diagonal=1)
    else:
        Q = A.clone()
        n = Q.size(0)
        indices = torch.arange(n, device=Q.device)
        A[indices, indices] = 0
    return Q


class ColoringSCIPSolver(BaseSCIPSolver):
    def __init__(self, edge_list, max_color=None, pre_solve=True):
        super().__init__(edge_list, pre_solve)
        self.max_color = max_color
        self.x = None  # x[v, k]
        self.y = None  # y[k]

    def _add_variables(self):
        max_degree = max(sum(1 for e in self.edge_list if v in e) for v in self.V)
        K_max = max_degree + 1 if self.max_color is None else self.max_color
        self.x = {(v, k): self.model.addVar(vtype="B", name=f"x_{v}_{k}") for v in self.V for k in range(1, K_max + 1)}
        self.y = {k: self.model.addVar(vtype="B", name=f"y_{k}") for k in range(1, K_max + 1)}

    def _add_constraints(self):
        for v in self.V:
            self.model.addCons(quicksum(self.x[v, k] for k in self.y) == 1, name=f"vertex_{v}_assignment")

        for idx, e in enumerate(self.edge_list):
            for k in self.y:
                self.model.addCons(quicksum(self.x[v, k] for v in e) <= len(e) - 1, name=f"edge_{idx}_color_{k}_constraint")

        for v in self.V:
            for k in self.y:
                self.model.addCons(self.x[v, k] <= self.y[k], name=f"color_{k}_usage_marker_{v}")

    def _set_objective(self):
        self.model.setObjective(quicksum(self.y[k] for k in self.y), "minimize")

    def _get_solution_metrics(self, sol):
        used_colors = sum(round(self.model.getSolVal(sol, var)) for var in self.y.values())
        return {"colors": used_colors}

    def _extract_solution(self, sol):
        color_assignment = {}
        for v in self.V:
            for k in self.y:
                if round(self.model.getSolVal(sol, self.x[v, k])) > 0.5:
                    color_assignment[v] = k
                    break
        return color_assignment


class _GraphTabuCol(BaseTabuCol):
    """For traditional graph coloring"""
    def __init__(self, edges, k, tabu_tenure=30, max_iter=1000):
        super().__init__(edges, k, tabu_tenure, max_iter)
        self.adj = defaultdict(list)
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def evaluate(self, solution):
        return sum(1 for u, v in self.edges if solution[u] == solution[v])

    def calculate_delta(self, u, old_color, new_color, solution):
        delta = 0
        for v in self.adj[u]:
            if solution[v] == old_color:
                delta -= 1
            if solution[v] == new_color:
                delta += 1
        return delta


class _HyperTabuCol(BaseTabuCol):
    """For hypergraph proper coloring"""

    def evaluate(self, solution):
        conflicts = 0
        for edge in self.edges:
            colors = set(solution[v] for v in edge)
            if len(colors) == 1:  # All same color -> conflict
                conflicts += 1
        return conflicts

    def calculate_delta(self, u, old_color, new_color, solution):
        delta = 0
        # Only check hyperedges containing u
        for edge_idx in self.vertex_to_edges[u]:
            edge = self.edges[edge_idx]

            # Check original state
            original_colors = [solution[v] for v in edge]
            was_conflict = len(set(original_colors)) == 1

            # Simulate color change
            new_colors = [new_color if v == u else solution[v] for v in edge]
            now_conflict = len(set(new_colors)) == 1

            # Update delta
            if was_conflict and not now_conflict:
                delta -= 1
            elif not was_conflict and now_conflict:
                delta += 1
        return delta


def coloring_tabu(
    type: Literal["graph", "hypergraph"], edges, init_k: Optional[int] = None, max_iter: int = 1000, max_time: Optional[float] = None
) -> Tuple[int, Optional[list]]:
    """Tabu search coloring with time tracking for each valid solution

    Coloring Tabu search searches from the upper bound `init_k` downwards
    step by step until it is impossible to find a solution that satisfies
    the constraints, at which point it stops.
    Args:
        type (str): Choose which task to perform.
        edges (list): Edge_list, allows for general graphs and hypergraphs
        init_k (int): Upper bound of different color numbers, optional but
                      still recommended to provide a reasonable value to
                      reduce the search range.
        max_iter (int): Each solution's maximum search times (seconds).
        max_time (int): Maximum total time (seconds).
    """

    is_hypergraph = type == "hypergraph"
    time_records: Dict[int, float] = {}

    print("\n=== Hypergraph Coloring with Tabu Search ===" if is_hypergraph else "\n=== Graph Coloring with Tabu Search ===")

    solver_class = _HyperTabuCol if is_hypergraph else _GraphTabuCol

    temp = solver_class(edges, 0)
    if is_hypergraph:
        max_degree = max(len(e) for e in edges)
    else:
        max_degree = max(len(temp.vertex_to_edges[v]) for v in range(temp.n))

    k = max_degree + 1 if init_k is None else init_k
    best_sol = None

    print(f"Max {'hyper' if is_hypergraph else ''}degree: {max_degree}, Initial color attempt: {k}")
    start_time = time.time()

    while k > 1:
        solver = solver_class(edges, k, max_iter=max_iter)
        solution, conflicts = solver.solve(max_time)

        if conflicts == 0:
            discovery_time = time.time() - start_time
            time_records[k] = discovery_time
            logger.info(f"\n✅ Valid {k}-coloring found at {discovery_time:.2f}s")
            best_sol = solution
            k -= 1
        else:
            logger.info(f"\n❌ No valid {k}-coloring found")
            break

    final_k = k + 1 if best_sol is not None else k
    print(f"\n=== Result: Minimum colors required = {final_k} ===")
    logger.info(f"\n=== Result: Minimum colors required = {final_k} ===")

    if time_records:
        print("\n=== Solution Discovery Timeline ===")
        logger.info("\n=== Solution Discovery Timeline ===")
        for k in sorted(time_records.keys(), reverse=True):
            time_str = f"{time_records[k]:.2f} seconds".rjust(10)
            print(f"k = {k} : discovered at {time_str}")
            logger.info(f"k = {k} : discovered at {time_str}")

    return final_k, best_sol


def coloring_evaluate(outs: torch.Tensor, graph, threshold=0.6):
    """
    Evaluate the effectiveness of the coloring scheme (supports regular graphs and hypergraphs)

    Args:
        outs: Model output tensor [num_nodes, num_colors]
        graph: The graph or hypergraph object
        threshold: Node color confidence threshold

    Returns:
        {
            "correct_edges": Number of edges that meet the condition,
            "total_edges": Total number of edges,
            "accuracy": accuracy rate,
            "color_distribution": color distribution,
            "not_converged": number of unconverged nodes
        }

    """
    nodes = sorted(graph.v)
    edges = graph.e[0]

    node2idx = {v: i for i, v in enumerate(nodes)}

    if outs.shape[0] != len(nodes):
        raise ValueError(f"outs contain{outs.shape[0]} vertices,  but {len(nodes)} vertices in graph")

    max_vals, max_indices = torch.max(outs, dim=1)
    outs_hard = torch.zeros_like(outs, dtype=torch.int)
    outs_hard.scatter_(1, max_indices.unsqueeze(1), 1)

    correct_edges = 0
    conflict_edges = 0
    not_converged = (max_vals < threshold).sum().item()
    color_distribution = outs_hard.sum(dim=0).cpu().numpy()

    for edge in edges:
        try:
            indices = [node2idx[v] for v in edge]
        except KeyError as e:
            raise ValueError(f"edge {edge} contains unknown v: {e}")

        edge_colors = outs_hard[indices]

        if len(edge) == 2:
            ok = edge_colors[0].argmax() != edge_colors[1].argmax()
        else:
            unique_colors = torch.unique(edge_colors.argmax(dim=1))
            ok = len(unique_colors) >= 2

        if ok:
            correct_edges += 1
        else:
            conflict_edges += 1

    total_edges = len(edges)
    accuracy = correct_edges / total_edges if total_edges > 0 else 0.0

    num_color_used = 0
    for i in color_distribution:
        if i != 0:
            num_color_used += 1

    print(f"+------------[Evaluation Result]------------+")
    print(
        f"Qualified Edge: {correct_edges}/{total_edges} ({accuracy:.1%})\n"
        f"Conflict Edge: {conflict_edges}\n"
        f"Color distribution:\n{color_distribution}\n"
        f"Not converged: {not_converged}\n"
        f"Num_Color_used: {num_color_used}"
    )
    print(f"+-------------------------------------------+")
    return {
        "correct_edges": correct_edges,
        "conflict_edges": conflict_edges,
        "total_edges": total_edges,
        "accuracy": accuracy,
        "color_distribution": color_distribution,
        "not_converged": not_converged,
        "num_color": num_color_used,
    }


def _coloring_scip(graph: dhg.Graph, time_limit=3600):
    """
    Test graph coloring, deprecated but still valid
    """

    class SolutionTracker(Eventhdlr):
        def __init__(self, y_vars):
            super().__init__()
            self.y_vars = y_vars

        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexec(self, event):
            current_sol = self.model.getBestSol()
            used_colors = sum(round(self.model.getSolVal(current_sol, var)) for var in self.y_vars.values())
            print(f"Current sol(colors): {used_colors}")

    model = Model("GraphColoring")
    if time_limit:
        model.setRealParam("limits/time", time_limit)
    V = list(graph.v)
    E = list(graph.e[0])

    max_degree = max(sum(1 for e in E if v in e) for v in V)
    K_max = max_degree + 1

    x = {(v, k): model.addVar(vtype="B", name=f"x_{v}_{k}") for v in V for k in range(1, K_max + 1)}
    y = {k: model.addVar(vtype="B", name=f"y_{k}") for k in range(1, K_max + 1)}

    # add constraints
    for v in V:
        model.addCons(quicksum(x[v, k] for k in range(1, K_max + 1)) == 1, name=f"assign_color_{v}")

    for u, v in E:
        for k in range(1, K_max + 1):
            model.addCons(x[u, k] + x[v, k] <= 1, name=f"edge_{u}_{v}_color_{k}")

    for v in V:
        for k in range(1, K_max + 1):
            model.addCons(x[v, k] <= y[k], name=f"color_used_{v}_{k}")

    # obj
    model.setObjective(quicksum(y[k] for k in y), "minimize")

    tracker = SolutionTracker(y)
    model.includeEventhdlr(tracker, "SolutionTracker", "Track colors")

    model.optimize()

    if model.getStatus() == "optimal":
        print(f"\nbest sol(colors): {int(model.getObjVal())}")
        sol = model.getBestSol()
        for v in V:
            for k in range(1, K_max + 1):
                if model.getSolVal(sol, x[v, k]) > 0.5:
                    print(f"Vertex_{v} → color_{k}")
                    break
    else:
        print("Non sol found")
