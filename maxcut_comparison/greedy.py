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

    return adj


def greedy_maxcut_naive(adj):
    """
    朴素贪心 MaxCut：
    按节点编号顺序依次决定每个点属于哪一侧，
    使其与“已经分配好的邻居”之间形成的割边尽可能多。
    """
    n = len(adj)
    part = [-1] * n 

    cnt0 = 0
    cnt1 = 0

    for u in range(n):
        gain_if_0 = 0
        gain_if_1 = 0

        for v in adj[u]:
            if part[v] == -1:
                continue
            if part[v] == 1:
                gain_if_0 += 1
            if part[v] == 0:
                gain_if_1 += 1

        if gain_if_0 > gain_if_1:
            part[u] = 0
            cnt0 += 1
        elif gain_if_1 > gain_if_0:
            part[u] = 1
            cnt1 += 1
        else:
            if cnt0 <= cnt1:
                part[u] = 0
                cnt0 += 1
            else:
                part[u] = 1
                cnt1 += 1

    return part


def evaluate_cut(graph, part):
    """
    统计 cut 值，并检查划分是否合法
    """
    if any(x not in (0, 1) for x in part):
        return 0, False

    cut_size = 0
    for edge in graph.e[0]:
        u, v = int(edge[0]), int(edge[1])
        if part[u] != part[v]:
            cut_size += 1

    return cut_size, True


def solve_one_dataset(dataset):
    """
    求解单个数据集
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj = build_adj_from_graph(graph)

    start = time.time()
    part = greedy_maxcut_naive(adj)
    elapsed = time.time() - start

    cut_size, is_valid = evaluate_cut(graph, part)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(graph.e[0]),
        "cut_size": cut_size,
        "is_valid": is_valid,
        "time_sec": elapsed,
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(graph.e[0])}")
    print(f"Cut size        : {cut_size}")
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
    parser = argparse.ArgumentParser(description="Naive Greedy baseline for MaxCut")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="maxcut_greedy_results.csv",
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
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph MaxCut。")
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
                "cut_size": None,
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
            "cut_size",
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