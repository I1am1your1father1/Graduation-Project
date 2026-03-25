from typing import Literal
import torch


def partitioning_construct_Q(graph):
    """Constructs graph partitioning as an `OH-QUBO` formulation requiring matrix Q.

    The partitioning problem is modeled as `Reduce_sum(X^T·Q⊙X)`, where:
    - `Reduce_sum` performs element-wise matrix summation
    - X represents the solution vector/matrix
    - Q is the problem-specific design matrix
        + In praph partitioning problem, diagonal elements of the Q matrix are the degrees of each vertex, with other elements being negative

    Nont:
        The diagonal of Q must handled separately in the loss function.
    """
    A = graph.A.to_dense().fill_diagonal_(0)
    diag = A.sum(dim=0)  # diagonal matrix constructed from the degree of each vertex
    q = diag - A  # Output, but please read `Note` in `partitioning_construct_Q`
    return (diag, -A)


def partitioning_evaluate(outs: torch.Tensor, graph, threshold=0.6):
    cuts = 0
    not_converged = 0
    
    max_values, max_indices = torch.max(outs, dim=1)
    outs_max = torch.zeros_like(outs, dtype=torch.float) # int
    outs_max.scatter_(1, max_indices.unsqueeze(1), 1)
    
    not_converged = (max_values <= threshold).sum().item()
    
    same_partition = outs_max.mm(outs_max.t())
    A = graph.A.to_dense().fill_diagonal_(0).float()

    cuts = (A * (1 - same_partition)).sum().item() / 2.0
    blce = outs_max.sum(dim=0).cpu().numpy()

    print(f"Cuts: {cuts}\n"
          f"Blce: {blce}\n"
          f"Not converged nodes: {not_converged}")
