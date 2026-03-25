import torch

def mis_construct_Q(graph):
    """
    Constructs graph mis as an `OH-QUBO` formulation requiring matrix Q.

    """
    n = graph.num_v
    p = 2
    A = graph.A.to_dense()
    Q = torch.zeros_like(A)
    Q = Q.fill_diagonal_(-1)

    Q = torch.where(A > 0, p / 2, Q)

    return Q


def mis_evaluate(outs: torch.Tensor, graph, threshold=0.7):
    """
    Evaluate the effectiveness of Maximum Independent Set (MIS) partitioning

    Args:
        outs: Model output tensor shape (num_nodes, 2) (one-hot probabilities for in-set/out-of-set)
        graph: The graph object
        threshold: Confidence threshold for node convergence

    Returns:
        {
            "ind_set_size": Size of independent set (high confidence in-set nodes),
            "violation_edges": Number of edges within the independent set,
            "total_nodes": Total number of nodes,
            "is_valid": Whether the set is a valid independent set (violation_edges == 0),
            "selected_nodes": Number of high confidence in-set nodes,
            "not_converged": Number of unconverged nodes
        }
    """
    nodes = sorted(graph.v)
    edges = graph.e[0]
    
    node2idx = {v: i for i, v in enumerate(nodes)}
    
    if outs.shape[0] != len(nodes):
        raise ValueError(f"Output contains {outs.shape[0]} nodes, but graph has {len(nodes)} nodes")
    
    # Get hard assignments with one-hot encoding
    if outs.dim() == 1 or outs.shape[1] == 1:
        # binary case (MIS)
        x = outs.squeeze()
        
        max_vals = x
        max_indices = (x > threshold).long()
        
        outs_hard = max_indices.unsqueeze(1)
    else:
        # one-hot case
        max_vals, max_indices = torch.max(outs, dim=1)
        outs_hard = torch.zeros_like(outs, dtype=torch.int)
        outs_hard.scatter_(1, max_indices.unsqueeze(1), 1)
    
    # Calculate convergence and set information
    not_converged = (max_vals < threshold).sum().item()
    
    # Identify high confidence in-set nodes (group 1)
    high_confidence_mask = (max_indices == 1) & (max_vals >= threshold)
    selected_nodes = high_confidence_mask.sum().item()
    
    # Track independent set nodes by index
    ind_set_indices = torch.where(high_confidence_mask)[0].tolist()
    
    # Create independent set node dictionary for fast lookup
    ind_set_dict = {node: True for node_idx in ind_set_indices for node in [nodes[node_idx]]}
    
    # Check for violation edges (edges within the independent set)
    violation_edges = 0
    for edge in edges:
        # Check if both nodes are in the independent set
        if all(node in ind_set_dict for node in edge):
            violation_edges += 1
    
    total_nodes = len(nodes)
    is_valid = violation_edges == 0
    
    print(f"+-------[Maximum Independent Set Evaluation]-------+")
    print(
        f"Independent Set Size: {selected_nodes}/{total_nodes}\n"
        f"Violation Edges: {violation_edges} {'(VALID)' if is_valid else '(INVALID)'}\n"
        f"Unconverged Nodes: {not_converged}"
    )
    print(f"+--------------------------------------------------+")
    
    return {
        "ind_set_size": selected_nodes,
        "violation_edges": violation_edges,
        "total_nodes": total_nodes,
        "is_valid": is_valid,
        "selected_nodes": selected_nodes,
        "not_converged": not_converged,
    }


def is_maximal_independent_set(solution, graph):
    # solution: shape (n,), 0/1 or probabilities
    selected = set(torch.where(solution > 0.7)[0].tolist())
    edges = graph.e[0]

    # 先检查是否为独立集
    for u, v in edges:
        if u in selected and v in selected:
            return False

    # 建邻接表
    neighbors = {i: set() for i in range(graph.num_v)}
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    # 再检查是否还能加入新点
    for v in range(graph.num_v):
        if v in selected:
            continue
        if all(u not in selected for u in neighbors[v]):
            return False

    return True