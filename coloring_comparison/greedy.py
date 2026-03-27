import argparse
import time
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.core import Datasets
from src.utils import from_file_to_graph


def build_adj_from_graph(graph):
    """
    将图转成邻接表
    """
    n = graph.num_v
    adj = [set() for _ in range(n)]

    for edge in graph.e[0]:
        if len(edge) != 2:
            raise ValueError(f"发现非二元边 {edge}，这不是普通 graph 数据。")
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    degree = [len(adj[u]) for u in range(n)]
    return adj, degree


def greedy_graph_coloring_naive(adj):
    """
    朴素贪心图着色：
    按节点编号顺序依次处理，每次选择当前可用的最小颜色
    """
    n = len(adj)
    colors = [-1] * n

    for u in range(n):
        used_colors = set()
        for v in adj[u]:
            if colors[v] != -1:
                used_colors.add(colors[v])

        c = 0
        while c in used_colors:
            c += 1
        colors[u] = c

    return colors


def evaluate_coloring(graph, colors):
    """
    评估着色结果：
    - num_colors: 实际使用颜色数
    - conflict_edges: 冲突边数
    - is_valid: 是否是合法着色
    """
    if any(c < 0 for c in colors):
        return None, None, False

    num_colors = max(colors) + 1 if len(colors) > 0 else 0

    conflict_edges = 0
    for edge in graph.e[0]:
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            continue
        if colors[u] == colors[v]:
            conflict_edges += 1

    is_valid = (conflict_edges == 0)
    return num_colors, conflict_edges, is_valid


def solve_one_dataset(dataset):
    """
    求解单个数据集
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, _ = build_adj_from_graph(graph)

    start = time.time()
    colors = greedy_graph_coloring_naive(adj)
    elapsed = time.time() - start

    num_colors, conflict_edges, is_valid = evaluate_coloring(graph, colors)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(graph.e[0]),
        "num_colors": num_colors,
        "conflict_edges": conflict_edges,
        "is_valid": is_valid,
        "time_sec": elapsed,
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(graph.e[0])}")
    print(f"Num colors      : {num_colors}")
    print(f"Conflict edges  : {conflict_edges}")
    print(f"Valid           : {is_valid}")
    print(f"Time (sec)      : {elapsed:.6f}")
    print("=" * 80)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Naive Greedy baseline for Graph Coloring")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="coloring_greedy_results.csv",
        help="保存的 csv 文件名",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset is None:
        datasets = get_graph_datasets()
    else:
        if not hasattr(Datasets, args.dataset):
            raise ValueError(f"未知数据集名: {args.dataset}")
        dataset = getattr(Datasets, args.dataset)
        if dataset.type != "graph":
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph coloring。")
        datasets = [dataset]

    all_results = []

    for dataset in datasets:
        try:
            result = solve_one_dataset(dataset)
            all_results.append(result)
        except Exception as e:
            print("=" * 80)
            print(f"[ERROR] Dataset {dataset.name} failed: {e}")
            print("=" * 80)
            all_results.append({
                "dataset_enum": dataset.name,
                "dataset_value": dataset.value,
                "dataset_type": dataset.type,
                "file_path": dataset.path,
                "num_nodes": None,
                "num_edges": None,
                "num_colors": None,
                "conflict_edges": None,
                "is_valid": False,
                "time_sec": None,
            })

    df = pd.DataFrame(all_results)

    df = df[
        [
            "dataset_enum",
            "dataset_value",
            "dataset_type",
            "file_path",
            "num_nodes",
            "num_edges",
            "num_colors",
            "conflict_edges",
            "is_valid",
            "time_sec",
        ]
    ]

    save_dir = ROOT_DIR / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / args.save_name
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n结果已保存到：")
    print(save_path)


if __name__ == "__main__":
    main()