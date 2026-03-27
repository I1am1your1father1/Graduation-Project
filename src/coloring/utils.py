from collections import defaultdict
import dhg
import time
import torch
import logging

from tqdm import tqdm
from typing import Dict, Literal, Optional, Tuple

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
