import argparse
import time
import random
from pathlib import Path
import sys

import numpy as np
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


def build_edge_list_from_graph(graph):
    """
    从图中提取无向边列表
    """

    edges = []

    for edge in graph.e[0]:

        if len(edge) != 2:
            raise ValueError(f"发现非二元边 {edge}，这不是普通 graph 数据。")

        u, v = int(edge[0]), int(edge[1])

        if u == v:
            continue

        if u < v:
            edges.append((u, v))
        else:
            edges.append((v, u))

    return sorted(list(set(edges)))


def random_greedy_balanced_partition(
    adj,
    num_parts=2,
    seed=None,
):
    """
    随机贪心 graph partitioning（min-cut）

    目标：
    - 最小化 cut edges
    - 保持 balanced partition
    """

    if seed is not None:
        random.seed(seed)

    n = len(adj)

    if num_parts <= 0:
        raise ValueError("num_parts 必须为正整数。")

    if num_parts > n:
        raise ValueError("num_parts 不能大于节点数。")

    part = [-1] * n

    base = n // num_parts
    extra = n % num_parts

    capacities = [
        base + (1 if i < extra else 0)
        for i in range(num_parts)
    ]

    part_sizes = [0] * num_parts

    order = list(range(n))

    random.shuffle(order)

    for u in order:

        candidate_parts = []

        min_inc_cut = None

        for p in range(num_parts):

            if part_sizes[p] >= capacities[p]:
                continue

            inc_cut = 0

            for v in adj[u]:

                if part[v] == -1:
                    continue

                if part[v] != p:
                    inc_cut += 1

            if min_inc_cut is None or inc_cut < min_inc_cut:

                min_inc_cut = inc_cut

                candidate_parts = [p]

            elif inc_cut == min_inc_cut:

                candidate_parts.append(p)

        if len(candidate_parts) == 0:
            raise RuntimeError("没有可用分区。")

        best_p = random.choice(candidate_parts)

        part[u] = best_p

        part_sizes[best_p] += 1

    return part


def evaluate_partition(
    num_nodes,
    edges,
    part,
    num_parts,
):
    """
    评估 graph partitioning 结果
    """

    if len(part) != num_nodes:
        return None, None, None, False

    if any((p < 0 or p >= num_parts) for p in part):
        return None, None, None, False

    cut_edges = 0

    for u, v in edges:

        if part[u] != part[v]:
            cut_edges += 1

    part_sizes = [0] * num_parts

    for u in range(num_nodes):

        part_sizes[part[u]] += 1

    balance_gap = max(part_sizes) - min(part_sizes)

    is_valid = True

    return cut_edges, part_sizes, balance_gap, is_valid


def solve_one_dataset(
    dataset,
    num_parts=2,
    num_trials=10,
):
    """
    对单个数据集进行多次随机贪心实验
    """

    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, _ = build_adj_from_graph(graph)

    edges = build_edge_list_from_graph(graph)

    cut_values = []

    total_time = 0.0

    best_cut_edges = None
    best_balance_gap = None
    best_part_sizes = None
    best_valid = False

    for trial in range(num_trials):

        start = time.time()

        part = random_greedy_balanced_partition(
            adj=adj,
            num_parts=num_parts,
            seed=trial,
        )

        elapsed = time.time() - start

        total_time += elapsed

        cut_edges, part_sizes, balance_gap, is_valid = evaluate_partition(
            num_nodes=graph.num_v,
            edges=edges,
            part=part,
            num_parts=num_parts,
        )

        cut_values.append(cut_edges)

        if best_cut_edges is None or cut_edges < best_cut_edges:

            best_cut_edges = cut_edges
            best_balance_gap = balance_gap
            best_part_sizes = part_sizes
            best_valid = is_valid

    avg_cut_edges = float(np.mean(cut_values))

    std_cut_edges = float(np.std(cut_values))

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(edges),
        "num_parts": num_parts,
        "num_trials": num_trials,
        "best_cut_edges": best_cut_edges,
        "avg_cut_edges": avg_cut_edges,
        "std_cut_edges": std_cut_edges,
        "best_balance_gap": best_balance_gap,
        "is_valid": best_valid,
        "avg_time_sec": total_time / num_trials,
        "part_sizes": (
            " ".join(map(str, best_part_sizes))
            if best_part_sizes is not None
            else ""
        ),
    }

    print("=" * 90)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(edges)}")
    print(f"Num parts       : {num_parts}")
    print(f"Trials          : {num_trials}")
    print(f"Best Cut        : {best_cut_edges}")
    print(f"Average Cut     : {avg_cut_edges:.4f}")
    print(f"Std Cut         : {std_cut_edges:.4f}")
    print(f"Part sizes      : {best_part_sizes}")
    print(f"Balance gap     : {best_balance_gap}")
    print(f"Valid           : {best_valid}")
    print(f"Avg Time (sec)  : {total_time / num_trials:.6f}")
    print("=" * 90)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Random Greedy baseline for Graph Partitioning"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )

    parser.add_argument(
        "--num_parts",
        type=int,
        default=2,
        help="划分的分区数量",
    )

    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="每个数据集随机运行次数",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="graph_partition_random_greedy_results.csv",
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
            raise ValueError(
                f"{args.dataset} 不是 graph 数据集。"
            )

        datasets = [dataset]

    all_results = []

    for dataset in datasets:

        try:

            result = solve_one_dataset(
                dataset=dataset,
                num_parts=args.num_parts,
                num_trials=args.num_trials,
            )

            all_results.append(result)

        except Exception as e:

            print("=" * 90)
            print(f"[ERROR] Dataset {dataset.name} failed: {e}")
            print("=" * 90)

            all_results.append({
                "dataset_enum": dataset.name,
                "dataset_value": dataset.value,
                "dataset_type": dataset.type,
                "file_path": dataset.path,
                "num_nodes": None,
                "num_edges": None,
                "num_parts": args.num_parts,
                "num_trials": args.num_trials,
                "best_cut_edges": None,
                "avg_cut_edges": None,
                "std_cut_edges": None,
                "best_balance_gap": None,
                "is_valid": False,
                "avg_time_sec": None,
                "part_sizes": "",
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
            "num_parts",
            "num_trials",
            "best_cut_edges",
            "avg_cut_edges",
            "std_cut_edges",
            "best_balance_gap",
            "part_sizes",
            "is_valid",
            "avg_time_sec",
        ]
    ]

    save_dir = ROOT_DIR / "results"

    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / args.save_name

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n结果已保存到：")
    print(save_path)


if __name__ == "__main__":
    main()