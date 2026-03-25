import os
import dhg
import torch
import logging
import pickle as pkl
from typing import Literal

from dhg import Hypergraph, Graph
from .core import get_current_seed

logger = logging.getLogger(__name__)


def from_file_to_graph(file_path: str, reset_vertex_index: bool = False, remove_self_loops: bool = True):
    """Read a graph from a file.

    Args:
        file_path (str): The path to the file containing the graph. File
            format specifications are described in the notes section below.
        reset_vertex_index (bool): Whether to reset the vertex index to start from
            0. If True, vertex indices will be reindexed starting at 0 regardless
            of their original numbering in the file.
        remove_self_loops (bool): Whether to remove self loops in the Graph

    Returns:
        graph: The constructed graph object

    Notes:
        The input file must adhere to the following structure:

            nums_edges nums_vertices     -> Optional, but must be left blank
            1 9
            1 17
            1 25
            1 33
            ...

        Where:
            - Line 1: Total number of edges and vertices (space-separated integers)
            - Subsequent lines:
                + Each line represents a edge
                + Space-separated integers represent vertices in the edge
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        lines = lines[1:]
        edges = []
        loops = []
        for line in lines:
            line = line.strip()
            edge = line.split()
            if edge[0] == edge[1]:
                loops.append(edge)
                continue
            edges.append(edge)
        if not remove_self_loops:
            edges = edges + loops
        all_vertices = sorted(set(vertex for edge in edges for vertex in edge))

        vertex_mapping = {old_vertex: new_vertex for new_vertex, old_vertex in enumerate(all_vertices, start=0)}
        new_edges = [[vertex_mapping[vertex] for vertex in edge] for edge in edges]
        print(f"INFO: {len(loops)} loops found")
        g = Graph(num_v=len(all_vertices), e_list=new_edges if reset_vertex_index else edges)

        return g


def from_hypergraph_to_graph_clique(hypergraph, remove_self_loops=True):
    tmp = dhg.Graph.from_hypergraph_clique(hypergraph)
    edge_list = tmp.e[0]
    edges = []
    loops = []
    for edge in edge_list:
        if edge[0] == edge[1]:
            loops.append(edge)
            continue
        edges.append(edge)
    if not remove_self_loops:
        edges = edges + loops
    g = Graph(num_v=hypergraph.num_v, e_list=edges)
    logger.info(f"INFO: {len(loops)} loops found")
    return g

def from_hypergraph_to_graph_hypergcn(hypergraph, remove_self_loops=True):
    """
    Construct a graph from a hypergraph with methods proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://arxiv.org/pdf/1809.02589.pdf>`_ paper
    """
    X = torch.randn((hypergraph.num_v, 2))
    torch.set_default_tensor_type(torch.FloatTensor)
    tmp = dhg.Graph.from_hypergraph_hypergcn(hypergraph.to("cpu"), X, remove_selfloop=remove_self_loops)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    edge_list = tmp.e[0]
    edges = []
    loops = []
    for edge in edge_list:
        if edge[0] == edge[1]:
            loops.append(edge)
            continue
        edges.append(edge)
    if not remove_self_loops:
        edges = edges + loops
    g = Graph(num_v=hypergraph.num_v, e_list=edges)
    logger.info(f"INFO: {len(loops)} loops found")
    return g


def from_file_to_hypergraph(file_path: str, reset_vertex_index=False) -> Hypergraph:
    """Read a hypergraph from a file.

    Args:
        file_path (str): The path to the file containing the hypergraph. File
            format specifications are described in the notes section below.
        reset_vertex_index (bool): Whether to reset the vertex index to start from
            0. If True, vertex indices will be reindexed starting at 0 regardless
            of their original numbering in the file.

    Returns:
        Hypergraph: The constructed hypergraph object with hyperedges and vertices
        as defined in the input file.

    Notes:
        The input file must adhere to the following structure:

            nums_edges nums_vertices             -> Optional, but must be left blank
            7008 8428 3566 38 1606 4146 5855 1014 7722 1739 7716 5817
            5056 4114 12483 10073 8546 10045
            11070 2289 4114 10073 1747 1628
            ...

        Where:
            - Line 1: Total number of hyperedges and vertices (space-separated integers)
            - Subsequent lines:
                * Each line represents a hyperedge
                * Space-separated integers represent vertices in the hyperedge
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        lines = lines[1:]
        edges = [list(map(int, line.split())) for line in lines]
        all_vertices = sorted(set(vertex for edge in edges for vertex in edge))

        vertex_mapping = {old_vertex: new_vertex for new_vertex, old_vertex in enumerate(all_vertices, start=0)}
        new_edges = [[vertex_mapping[vertex] for vertex in edge] for edge in edges]

        hg = Hypergraph(num_v=len(all_vertices), e_list=new_edges if reset_vertex_index else edges)

        print(hg)
        return hg


def from_file_to_hypergraph_(file_path: str, reset_vertex_index: bool = False) -> Hypergraph:
    """Read a hypergraph from a file with support for string vertex names.

    Args:
        file_path: Path to the hypergraph file
        reset_vertex_index: Whether to reindex vertices starting from 0

    Returns:
        Hypergraph with parsed vertices and hyperedges
    """
    with open(file_path, "r") as file:
        # Read and preprocess lines
        lines = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]

        # Skip optional first line with two numbers
        if lines and len(lines[0].split()) == 2:
            lines = lines[1:]

        # Parse edges as strings
        edges = [line.split() for line in lines]

        # Create sorted vertex list with numeric优先排序
        def sort_key(v):
            try:
                # 数字优先按数值排序
                return (0, int(v))
            except ValueError:
                # 非数字按字母排序
                return (1, v)

        all_vertices = sorted({v for edge in edges for v in edge}, key=sort_key)

        # Create mapping if reindexing needed
        vertex_mapping = {old: idx for idx, old in enumerate(all_vertices)}

        # Process edges based on reindex flag
        processed_edges = []
        for edge in edges:
            if reset_vertex_index:
                processed_edges.append([vertex_mapping[v] for v in edge])
            else:
                processed_edges.append(edge)

        # Create hypergraph instance
        return Hypergraph(num_v=len(all_vertices), e_list=processed_edges)


def from_pickle_to_hypergraph(dataset: str) -> Hypergraph:
    """Read a hypergraph from a pickle file.

    Args:
        file_path (str): The path to the pickle file containing the hypergraph.

    Returns:
        Hypergraph: The hypergraph read from the pickle file.

    Examples:
        >>> from src import from_pickle_to_hypergraph
        >>> hg = from_pickle_to_hypergraph("data/test_hypergraph.pkl")
        >>> print(hg.num_vertices())
        128
    """
    data_path = os.path.join(dataset)

    with open(data_path, "rb") as f:
        H = pkl.load(f)
    l: dict[int, list] = {}
    for i, j in zip(H[0], H[1]):
        i, j = i.item(), j.item()
        if l.get(j):
            l[j].append(i)
        else:
            l[j] = [i]
    sorted_l = {k: v for k, v in sorted(l.items(), key=lambda item: item[0])}
    num_v = H[0].max().item() + 1
    e_list = list(sorted_l.values())
    return Hypergraph(num_v, e_list, merge_op="mean")


def edge_weight(edge_list, e=2.0) -> torch.Tensor:
    if not isinstance(edge_list, torch.Tensor):
        edge_list = torch.tensor(edge_list, dtype=torch.long)

    src_nodes = edge_list[:, 0]
    dst_nodes = edge_list[:, 1]

    all_nodes = torch.cat([src_nodes, dst_nodes])
    degrees = torch.bincount(all_nodes)

    src_degrees = degrees[src_nodes]
    dst_degrees = degrees[dst_nodes]

    return (src_degrees.float() + dst_degrees.float()) / e


def generate_data(
    type: Literal["graph", "hypergraph"], v, e, seed=None, hyper_method: Literal["uniform", "low_order_first", "high_order_first"] = "low_order_first"
):
    """Generate random graph or hypergraph. More way to generate data see [DHG.random](https://deephypergraph.readthedocs.io/en/latest/api/random.html?highlight=random)

    Args:
        type (Literal["graph", "hypergraph"]): :)
        v (int): Number of vertices in the structure
        e (int): Number of edges/hyperedges to generate
        seed (int, optional): Random seed for reproducibility. Uses current module seed if None.
        hyper_method (Literal["uniform", "low_order_first", "high_order_first"]):
            Hyperedge generation method (only for hypergraphs). Defaults to "low_order_first".

    Note:
        Under the same seed, the generated graph structure is the same.
    """

    dhg.random.set_seed(seed if seed is not None else get_current_seed())
    if type == "graph":
        data = dhg.random.graph_Gnm(v, e)
    elif type == "hypergraph":
        data = dhg.random.hypergraph_Gnm(v, e, method=hyper_method)
    else:
        raise ValueError("Param type error")
    return data
