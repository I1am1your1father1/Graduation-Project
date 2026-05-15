import argparse
import time
import random
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


def random_greedy_graph_coloring(adj, seed=None):
    """
    Random Greedy 图着色：

    每次运行随机打乱节点访问顺序；
    按随机顺序依次处理节点；
    每个节点选择当前可用的最小颜色。
    """
    rng = random.Random(seed)

    n = len(adj)
    colors = [-1] * n

    node_order = list(range(n))
    rng.shuffle(node_order)

    for u in node_order:
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


def solve_one_dataset_one_run(dataset, run_id, seed):
    """
    单个数据集运行一次 Random Greedy
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, _ = build_adj_from_graph(graph)

    start = time.time()
    colors = random_greedy_graph_coloring(adj, seed=seed)
    elapsed = time.time() - start

    num_colors, conflict_edges, is_valid = evaluate_coloring(graph, colors)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "run": run_id,
        "seed": seed,
        "num_nodes": graph.num_v,
        "num_edges": len(graph.e[0]),
        "num_colors": num_colors,
        "conflict_edges": conflict_edges,
        "is_valid": is_valid,
        "time_sec": elapsed,
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Run             : {run_id}")
    print(f"Seed            : {seed}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(graph.e[0])}")
    print(f"Num colors      : {num_colors}")
    print(f"Conflict edges  : {conflict_edges}")
    print(f"Valid           : {is_valid}")
    print(f"Time (sec)      : {elapsed:.6f}")
    print("=" * 80)

    return result


def summarize_results(df):
    """
    汇总每个数据集的 Random Greedy 图着色结果：

    图着色问题中，颜色数越少越好；
    若存在合法结果，则优先在合法结果中选择颜色数最少的结果。
    """
    summary = []

    for dataset_name, group in df.groupby("dataset_enum"):
        success_group = group[group["num_colors"].notna()].copy()

        if len(success_group) == 0:
            summary.append({
                "dataset_enum": dataset_name,
                "dataset_value": None,
                "num_nodes": None,
                "num_edges": None,
                "best_num_colors": None,
                "best_run": None,
                "best_seed": None,
                "best_conflict_edges": None,
                "best_is_valid": False,
                "avg_num_colors": None,
                "std_num_colors": None,
                "avg_conflict_edges": None,
                "avg_time_sec": None,
                "std_time_sec": None,
                "valid_runs": 0,
                "success_runs": 0,
            })
            continue

        valid_group = success_group[success_group["is_valid"] == True].copy()

        if len(valid_group) > 0:
            best_row = valid_group.loc[valid_group["num_colors"].idxmin()]
        else:
            success_group["invalid_score"] = (
                success_group["conflict_edges"] * (success_group["num_nodes"] + 1)
                + success_group["num_colors"]
            )
            best_row = success_group.loc[success_group["invalid_score"].idxmin()]

        summary.append({
            "dataset_enum": dataset_name,
            "dataset_value": best_row["dataset_value"],
            "num_nodes": int(success_group["num_nodes"].iloc[0]),
            "num_edges": int(success_group["num_edges"].iloc[0]),

            "best_num_colors": int(best_row["num_colors"]),
            "best_run": int(best_row["run"]),
            "best_seed": int(best_row["seed"]),
            "best_conflict_edges": int(best_row["conflict_edges"]),
            "best_is_valid": bool(best_row["is_valid"]),

            "avg_num_colors": float(success_group["num_colors"].mean()),
            "std_num_colors": float(success_group["num_colors"].std(ddof=0)),
            "avg_conflict_edges": float(success_group["conflict_edges"].mean()),

            "avg_time_sec": float(success_group["time_sec"].mean()),
            "std_time_sec": float(success_group["time_sec"].std(ddof=0)),

            "valid_runs": int(success_group["is_valid"].sum()),
            "success_runs": int(len(success_group)),
        })

    return pd.DataFrame(summary)


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Random Greedy baseline for Graph Coloring")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="每个数据集重复运行次数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="随机种子起始值",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="coloring_random_greedy_details.csv",
        help="详细结果 csv 文件名",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="coloring_random_greedy_summary.csv",
        help="汇总结果 csv 文件名",
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

    for dataset_idx, dataset in enumerate(datasets):
        for run_id in range(1, args.num_runs + 1):
            seed = args.seed + dataset_idx * 1000 + run_id - 1

            try:
                result = solve_one_dataset_one_run(
                    dataset=dataset,
                    run_id=run_id,
                    seed=seed,
                )
                all_results.append(result)

            except Exception as e:
                print("=" * 80)
                print(f"[ERROR] Dataset {dataset.name}, Run {run_id} failed: {e}")
                print("=" * 80)

                all_results.append({
                    "dataset_enum": dataset.name,
                    "dataset_value": dataset.value,
                    "dataset_type": dataset.type,
                    "file_path": dataset.path,
                    "run": run_id,
                    "seed": seed,
                    "num_nodes": None,
                    "num_edges": None,
                    "num_colors": None,
                    "conflict_edges": None,
                    "is_valid": False,
                    "time_sec": None,
                })

    df_detail = pd.DataFrame(all_results)

    df_detail = df_detail[
        [
            "dataset_enum",
            "dataset_value",
            "dataset_type",
            "file_path",
            "run",
            "seed",
            "num_nodes",
            "num_edges",
            "num_colors",
            "conflict_edges",
            "is_valid",
            "time_sec",
        ]
    ]

    df_summary = summarize_results(df_detail)

    save_dir = ROOT_DIR / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    detail_path = save_dir / args.save_name
    summary_path = save_dir / args.summary_name

    df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n详细结果已保存到：")
    print(detail_path)

    print("\n汇总结果已保存到：")
    print(summary_path)

    print("\n汇总结果：")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()