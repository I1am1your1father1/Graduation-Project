import torch

def mds_evaluate(outs: torch.Tensor, graph, threshold=0.7):
    """
    Evaluate the performance of MDS algorithm.

    Args:
        outs (torch.Tensor): The output of MDS algorithm.
        graph (nx.Graph): The input graph.
        threshold (float, optional): The threshold for dominating set. Defaults to 0.7.

    Returns:
        {
            "dominating_set_size": Number of dominating set nodes,
            "total_nodes": Total number of nodes in the graph,
            "coverage": The ratio of dominating set nodes to total nodes,
            "is_valid": Whether the solution is valid,
            "not_converged": Number of unconverged nodes,
            "coverage_breakdown": (Number of nodes in DS, Number of nodes covered by neighbors)
        }
    """
    nodes = sorted(graph.v)
    edges = graph.e[0]

    node2idx = {v: i for i, v in enumerate(nodes)}

    if outs.shape[0] != len(nodes):
        raise ValueError(f"Output contains {outs.shape[0]} nodes, but graph has {len(nodes)} nodes")
    if outs.dim() != 2 or outs.size(1) != 2:
        raise ValueError(f"For OH-PUBO MDS, outs must have shape [V, 2], but got {tuple(outs.shape)}")

    # selected nodes = 1
    pred = torch.argmax(outs, dim=1)
    conf = outs.max(dim=1).values
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