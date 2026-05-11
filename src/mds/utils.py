import torch


def _get_edges(graph):
    edges_raw = graph.e[0]

    if isinstance(edges_raw, torch.Tensor) and edges_raw.dim() == 2 and edges_raw.shape[0] == 2:
        edges = edges_raw.t().tolist()
    else:
        edges = edges_raw

    return edges


def mds_construct_B(graph):
    """
    Construct closed-neighborhood matrix B for MDS-QUBO.

    B[v, u] = 1 means node u is in the closed neighborhood of node v.
    """
    nodes = sorted(graph.v)
    edges = _get_edges(graph)

    node2idx = {v: i for i, v in enumerate(nodes)}

    B = torch.zeros((len(nodes), len(nodes)))

    # closed neighborhood includes self
    for node in nodes:
        idx = node2idx[node]
        B[idx, idx] = 1.0

    # add neighbors
    for edge in edges:
        u, v = edge
        iu = node2idx[u]
        iv = node2idx[v]

        B[iu, iv] = 1.0
        B[iv, iu] = 1.0

    return B


def mds_construct_Q(graph, penalty=2.0):
    """
    Construct an explicit Q matrix for MDS-QUBO.

    The loss:
        sum_i x_i + penalty * sum_v (1 - sum_{u in N[v]} x_u)^2

    Ignoring the constant term, this can be written as:
        x^T Q x

    This function is mainly provided for compatibility with run_qubo.
    The training loss can also directly use B for clearer obj/cons logging.
    """
    B = mds_construct_B(graph).float()
    n = B.shape[0]

    ones = torch.ones(n)

    # quadratic part: penalty * x^T B^T B x
    Q = penalty * (B.t() @ B)

    # linear part: sum_i x_i - 2 * penalty * 1^T B x
    # represented on diagonal because x_i^2 = x_i for binary variables
    linear_diag = torch.ones(n) - 2 * penalty * (B.t() @ ones)

    Q = Q + torch.diag(linear_diag)

    return Q


def mds_evaluate(outs: torch.Tensor, graph, threshold=0.7, decision_threshold=0.5):
    """
    Evaluate the performance of MDS-QUBO algorithm.

    Args:
        outs: Model output tensor.
              If shape is [V, 1], outs[:, 0] is selected probability.
              If shape is [V, 2], outs[:, 1] is selected probability.
        graph: The input graph.
        threshold: Confidence threshold for node convergence.
        decision_threshold: Decision threshold for selected nodes.

    Returns:
        {
            "is_dominating_set": Whether the solution is valid,
            "dominating_set_size": Number of dominating set nodes,
            "dominating_set_nodes": Selected nodes,
            "uncovered_nodes": Nodes not dominated,
            "coverage_ratio": Coverage ratio,
            "not_converged": Number of unconverged nodes
        }
    """
    nodes = sorted(graph.v)
    edges = _get_edges(graph)

    node2idx = {v: i for i, v in enumerate(nodes)}

    if outs.shape[0] != len(nodes):
        raise ValueError(f"Output contains {outs.shape[0]} nodes, but graph has {len(nodes)} nodes")

    if outs.dim() == 1:
        selected_prob = outs
        pred = (selected_prob >= decision_threshold).long()
        conf = torch.maximum(selected_prob, 1 - selected_prob)

    elif outs.dim() == 2 and outs.size(1) == 1:
        selected_prob = outs[:, 0]
        pred = (selected_prob >= decision_threshold).long()
        conf = torch.maximum(selected_prob, 1 - selected_prob)

    elif outs.dim() == 2 and outs.size(1) == 2:
        pred = torch.argmax(outs, dim=1)
        conf = outs.max(dim=1).values

    else:
        raise ValueError(f"Unsupported outs shape for MDS-QUBO: {tuple(outs.shape)}")

    not_converged = conf < threshold

    adjacency = {node: set() for node in nodes}

    for edge in edges:
        u, v = edge
        adjacency[u].add(v)
        adjacency[v].add(u)

    covered = set()
    in_ds = set()

    for node in nodes:
        idx = node2idx[node]
        if pred[idx].item() == 1:
            in_ds.add(node)

    for node in in_ds:
        covered.add(node)
        covered.update(adjacency[node])

    uncovered_nodes = []

    for node in nodes:
        if node not in covered:
            uncovered_nodes.append(node)

    is_valid = len(uncovered_nodes) == 0
    dominating_set_sorted = sorted(in_ds)

    print(f"+-------[Min Dominating Set Evaluation]-------+")
    print("total_nodes:", len(nodes))
    print("in_ds:", len(in_ds))
    print("DS nodes:", dominating_set_sorted)
    print("covered:", len(covered))
    print("rating:", len(covered) / len(nodes))
    print("uncovered_nodes:", len(uncovered_nodes))
    print("not_converged:", not_converged.sum().item())
    print(f"+---------------------------------------------+")

    return {
        "is_dominating_set": is_valid,
        "dominating_set_size": len(in_ds),
        "dominating_set_nodes": dominating_set_sorted,
        "uncovered_nodes": uncovered_nodes,
        "coverage_ratio": len(covered) / len(nodes),
        "not_converged": not_converged.sum().item(),
    }