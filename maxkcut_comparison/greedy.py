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


def build_adj_and_edges_from_graph(graph):
    """
    将图转成邻接表和边列表
    """
    n = graph.num_v
    adj = [set() for _ in range(n)]
    edge_set = set()

    for edge in graph.e[0]:

        if len(edge) != 2:
            raise ValueError(f"发现非二元边 {edge}，这不是普通 graph 数据。")

        u, v = int(edge[0]), int(edge[1])

        if u == v:
            continue

        if u > v:
            u, v = v, u

        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))

    edge_list = sorted(list(edge_set))

    return adj, edge_list


def random_greedy_maxkcut(adj, k=3, seed=None):
    """
    随机贪心 Max-k-Cut：

    按随机顺序依次处理节点，
    将每个节点分配到当前能够带来最大 cut 增益的分区。
    如果多个分区增益相同，则从这些分区中随机选择。
    """

    if seed is not None:
        random.seed(seed)

    if k <= 1:
        raise ValueError("k 必须大于 1。")

    n = len(adj)

    part = [-1] * n

    order = list(range(n))

    random.shuffle(order)

    for u in order:

        gains = [0 for _ in range(k)]

        for v in adj[u]:

            if part[v] == -1:
                continue

            for c in range(k):

                if part[v] != c:
                    gains[c] += 1

        max_gain = max(gains)

        best_colors = [
            c for c in range(k)
            if gains[c] == max_gain
        ]

        chosen_color = random.choice(best_colors)

        part[u] = chosen_color

    return part


def evaluate_maxkcut(edge_list, part, k):
    """
    统计 Max-k-Cut 的 cut 值，并检查划分是否合法
    """

    if any((x < 0 or x >= k) for x in part):
        return 0, False

    cut_size = 0

    for u, v in edge_list:

        if part[u] != part[v]:
            cut_size += 1

    return cut_size, True


def solve_one_dataset(dataset, k=3, num_trials=10):
    """
    对单个数据集进行十次随机贪婪 Max-k-Cut 实验
    """

    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, edge_list = build_adj_and_edges_from_graph(graph)

    trial_results = []

    cut_sizes = []

    total_time = 0.0

    best_cut_size = -1
    best_valid = False
    best_trial = None

    for trial in range(num_trials):

        start = time.time()

        part = random_greedy_maxkcut(
            adj,
            k=k,
            seed=trial,
        )

        elapsed = time.time() - start

        total_time += elapsed

        cut_size, is_valid = evaluate_maxkcut(
            edge_list,
            part,
            k,
        )

        cut_sizes.append(cut_size)

        trial_result = {
            "dataset_enum": dataset.name,
            "dataset_value": dataset.value,
            "dataset_type": dataset.type,
            "file_path": dataset.path,
            "num_nodes": graph.num_v,
            "num_edges": len(edge_list),
            "k": k,
            "trial": trial,
            "seed": trial,
            "cut_size": cut_size,
            "is_valid": is_valid,
            "time_sec": elapsed,
        }

        trial_results.append(trial_result)

        if cut_size > best_cut_size:
            best_cut_size = cut_size
            best_valid = is_valid
            best_trial = trial

    avg_cut_size = float(np.mean(cut_sizes))

    std_cut_size = float(np.std(cut_sizes))

    summary_result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(edge_list),
        "k": k,
        "num_trials": num_trials,
        "best_trial": best_trial,
        "best_cut_size": best_cut_size,
        "avg_cut_size": avg_cut_size,
        "std_cut_size": std_cut_size,
        "is_valid": best_valid,
        "avg_time_sec": total_time / num_trials,
        "total_time_sec": total_time,
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(edge_list)}")
    print(f"k               : {k}")
    print(f"Trials          : {num_trials}")
    print(f"Best Trial      : {best_trial}")
    print(f"Best Cut        : {best_cut_size}")
    print(f"Average Cut     : {avg_cut_size:.4f}")
    print(f"Std Cut         : {std_cut_size:.4f}")
    print(f"Valid           : {best_valid}")
    print(f"Avg Time (sec)  : {total_time / num_trials:.6f}")
    print(f"Total Time(sec) : {total_time:.6f}")
    print("=" * 80)

    return summary_result, trial_results


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():

    parser = argparse.ArgumentParser(
        description="Random Greedy baseline for Max-k-Cut"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Max-k-Cut 中的分区数量 k",
    )

    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="每个数据集随机运行次数，默认每个数据集运行十次",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="maxkcut_random_greedy_results.csv",
        help="汇总结果保存的 csv 文件名",
    )

    parser.add_argument(
        "--detail_save_name",
        type=str,
        default="maxkcut_random_greedy_detail_results.csv",
        help="每次运行详细结果保存的 csv 文件名",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.k <= 1:
        raise ValueError("k 必须大于 1。")

    datasets = get_graph_datasets()

    all_summary_results = []
    all_trial_results = []

    for dataset in datasets:

        try:

            summary_result, trial_results = solve_one_dataset(
                dataset,
                k=args.k,
                num_trials=args.num_trials,
            )

            all_summary_results.append(summary_result)
            all_trial_results.extend(trial_results)

        except Exception as e:

            print("=" * 80)
            print(f"[ERROR] Dataset {dataset.name} failed: {e}")
            print("=" * 80)

            all_summary_results.append({
                "dataset_enum": dataset.name,
                "dataset_value": dataset.value,
                "dataset_type": dataset.type,
                "file_path": dataset.path,
                "num_nodes": None,
                "num_edges": None,
                "k": args.k,
                "num_trials": args.num_trials,
                "best_trial": None,
                "best_cut_size": None,
                "avg_cut_size": None,
                "std_cut_size": None,
                "is_valid": False,
                "avg_time_sec": None,
                "total_time_sec": None,
            })

            all_trial_results.append({
                "dataset_enum": dataset.name,
                "dataset_value": dataset.value,
                "dataset_type": dataset.type,
                "file_path": dataset.path,
                "num_nodes": None,
                "num_edges": None,
                "k": args.k,
                "trial": None,
                "seed": None,
                "cut_size": None,
                "is_valid": False,
                "time_sec": None,
            })

    summary_df = pd.DataFrame(all_summary_results)

    summary_df = summary_df[
        [
            "dataset_enum",
            "dataset_value",
            "dataset_type",
            "file_path",
            "num_nodes",
            "num_edges",
            "k",
            "num_trials",
            "best_trial",
            "best_cut_size",
            "avg_cut_size",
            "std_cut_size",
            "is_valid",
            "avg_time_sec",
            "total_time_sec",
        ]
    ]

    detail_df = pd.DataFrame(all_trial_results)

    detail_df = detail_df[
        [
            "dataset_enum",
            "dataset_value",
            "dataset_type",
            "file_path",
            "num_nodes",
            "num_edges",
            "k",
            "trial",
            "seed",
            "cut_size",
            "is_valid",
            "time_sec",
        ]
    ]

    save_dir = ROOT_DIR / "results"

    save_dir.mkdir(parents=True, exist_ok=True)

    summary_save_path = save_dir / args.save_name
    detail_save_path = save_dir / args.detail_save_name

    summary_df.to_csv(
        summary_save_path,
        index=False,
        encoding="utf-8-sig",
    )

    detail_df.to_csv(
        detail_save_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n汇总结果已保存到：")
    print(summary_save_path)

    print("\n每次运行详细结果已保存到：")
    print(detail_save_path)


if __name__ == "__main__":
    main()