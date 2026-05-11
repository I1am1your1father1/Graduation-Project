import time
import torch
from pyscipopt import quicksum, Model, SCIP_PARAMSETTING, SCIP_EVENTTYPE, Eventhdlr
from ..core import BaseSCIPSolver


def maxcut_construct_Q(graph):
    """Constructs graph max-cut as an `OH-QUBO` formulation requiring matrix Q.

    The coloring problem is modeled as `Reduce_sum(X^T·Q⊙X)`, where:
    - `Reduce_sum` performs element-wise matrix summation.
    - X represents the solution vector/matrix.
    - Q is the problem-specific design matrix, for max-cut
        * In undirected graph max-cut problem, the Q is
            + diagonal elements with the negative values of vertices' degrees
            + Upper triangular matrix multiplication of the adjacency matrix by element 2
        
    """
    A = graph.A.to_dense().clone()
    A.fill_diagonal_(0)

    Q = torch.zeros_like(A)
    Q -= torch.diag(A.sum(dim=0))
    Q += 2 * A.triu(1)
    
    return Q

def maxcut_evaluate(outs: torch.Tensor, graph, threshold=0.7, decision_threshold=0.5):
    """
    Evaluate the effectiveness of Max-Cut partitioning

    Args:
        outs: Model output tensor.
              If shape is (num_nodes, 1), outs[:, 0] is treated as the probability of group 1.
              If shape is (num_nodes, 2), outs is treated as one-hot probabilities for two groups.
        graph: The graph object
        threshold: Confidence threshold for node convergence
        decision_threshold: Decision threshold for binary sigmoid output

    Returns:
        {
            "cut_edges": Number of successfully cut edges,
            "total_edges": Total number of edges,
            "accuracy": Cut accuracy rate,
            "group_distribution": Tuple of (group0_count, group1_count),
            "not_converged": Number of unconverged nodes
        }
    """
    nodes = sorted(graph.v)
    edges_raw = graph.e[0]

    if isinstance(edges_raw, torch.Tensor) and edges_raw.dim() == 2 and edges_raw.shape[0] == 2:
        edges = edges_raw.t().tolist()
    else:
        edges = edges_raw

    node2idx = {v: i for i, v in enumerate(nodes)}

    if outs.shape[0] != len(nodes):
        raise ValueError(f"Output contains {outs.shape[0]} nodes, but graph has {len(nodes)} nodes")

    # Get hard assignments
    if outs.dim() == 1 or outs.shape[1] == 1:
        # Binary sigmoid case:
        # outs[:, 0] >= decision_threshold -> group 1
        # outs[:, 0] <  decision_threshold -> group 0
        probs = outs.view(-1)

        max_indices = (probs >= decision_threshold).long()

        # For binary sigmoid output, confidence means being close to 0 or 1.
        max_vals = torch.maximum(probs, 1 - probs)

        group0_count = int((max_indices == 0).sum().item())
        group1_count = int((max_indices == 1).sum().item())
        group_distribution = (group0_count, group1_count)

    else:
        # One-hot / two-class probability case
        max_vals, max_indices = torch.max(outs, dim=1)

        outs_hard = torch.zeros_like(outs, dtype=torch.int)
        outs_hard.scatter_(1, max_indices.unsqueeze(1), 1)

        group_counts = outs_hard.sum(dim=0).cpu().numpy().astype(int)

        # Avoid index error when one group has no node
        group0_count = int(group_counts[0]) if len(group_counts) > 0 else 0
        group1_count = int(group_counts[1]) if len(group_counts) > 1 else 0
        group_distribution = (group0_count, group1_count)

    # Calculate convergence
    not_converged = (max_vals < threshold).sum().item()

    # Count cut edges
    cut_edges = 0
    for edge in edges:
        try:
            # Get indices for both nodes in the edge
            indices = [node2idx[int(v)] for v in edge]
        except KeyError as e:
            raise ValueError(f"Edge {edge} contains unknown node: {e}")

        # Get group assignments for both nodes
        group_assignments = max_indices[indices]

        # Check if nodes are in different groups
        if group_assignments[0] != group_assignments[1]:
            cut_edges += 1

    total_edges = len(edges)
    accuracy = cut_edges / total_edges if total_edges > 0 else 0.0

    print(f"+------------[MaxCut Evaluation]------------+")
    print(
        f"Cut Edges: {cut_edges}/{total_edges} ({accuracy:.1%})\n"
        f"Group Distribution: {group_distribution[0]} vs {group_distribution[1]}\n"
        f"Unconverged Nodes: {not_converged}"
    )
    print(f"+--------------------------------------------+")

    return {
        "cut_edges": cut_edges,
        "total_edges": total_edges,
        "accuracy": accuracy,
        "group_distribution": group_distribution,
        "not_converged": not_converged,
    }


class MaxCutSCIPSolver(BaseSCIPSolver):
    def __init__(self, edge_list, pre_solve=True):
        super().__init__(edge_list, pre_solve)
        self.x = None  # Vertices {0,1}
        self.a = None  # a[e] `v` in edge `e` all 0 or not 
        self.b = None  # b[e] `v` in edge `e` all 1 or not 
        self.z = None  # z[e] `e` is a cut
        self.edges = None

    def _add_variables(self):
        self.x = {v: self.model.addVar(vtype="B", name=f"x_{v}") for v in self.V}
        unique_edges = {tuple(sorted(e)) for e in self.edge_list}
        self.edges = list(unique_edges)
        self.a = {}
        self.b = {}
        self.z = {}
        for e in self.edges:
            e_str = '_'.join(map(str, e))
            self.a[e] = self.model.addVar(vtype="B", name=f"a_{e_str}")
            self.b[e] = self.model.addVar(vtype="B", name=f"b_{e_str}")
            self.z[e] = self.model.addVar(vtype="B", name=f"z_{e_str}")

    def _add_constraints(self):
        for e in self.edges:
            # constrain a[e] `v` in edge `e` all 0 or not 
            for v in e:
                self.model.addCons(self.x[v] <= 1 - self.a[e], name=f"a_{e}_{v}_leq")
            sum_x = quicksum(self.x[v] for v in e)
            self.model.addCons(sum_x >= 1 - self.a[e], name=f"a_{e}_sum_geq")

            # constrain b[e] `v` in edge `e` all 1 or not 
            for v in e:
                self.model.addCons(self.x[v] >= self.b[e], name=f"b_{e}_{v}_geq")
            sum_neg_x = quicksum(1 - self.x[v] for v in e)
            self.model.addCons(sum_neg_x >= 1 - self.b[e], name=f"b_{e}_sum_neg_geq")

            # constrain z[e] <= 1 - a[e] - b[e]
            self.model.addCons(self.z[e] <= 1 - (self.a[e] + self.b[e]), name=f"z_{e}_constraint")

    def _set_objective(self):
        total_cut = quicksum(self.z[e] for e in self.edges)
        self.model.setObjective(total_cut, "maximize")
        
    def _get_solution_metrics(self, sol):
        cut_count = sum(round(self.model.getSolVal(sol, self.z[e])) for e in self.edges)
        return {"cut_edges": cut_count}

    def _extract_solution(self, sol):
        solution = {v: round(self.model.getSolVal(sol, self.x[v])) for v in self.V}
        return solution