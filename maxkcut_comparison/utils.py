from dhg import Graph


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