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


def random_greedy_mds(adj, seed=None):
    """
    Random Greedy 求解最小支配集：

    每一步从能够新支配至少一个“当前未被支配节点”的候选顶点中随机选择一个点。
    """
    rng = random.Random(seed)

    n = len(adj)

    # closed_neighborhood[u] = {u} U N(u)
    closed_neighborhood = [set([u]) | adj[u] for u in range(n)]

    undominated = set(range(n))
    selected = [False] * n

    while undominated:
        candidates = []

        for u in range(n):
            gain = len(closed_neighborhood[u] & undominated)
            if gain > 0:
                candidates.append(u)

        if len(candidates) == 0:
            # 理论上一般不会进入这里，除非图或邻接关系异常
            best_u = rng.choice(list(undominated))
        else:
            best_u = rng.choice(candidates)

        selected[best_u] = True
        undominated -= closed_neighborhood[best_u]

    return selected


def evaluate_mds(graph, selected):
    """
    评估支配集结果：
    - mds_size: 选中的点数
    - undominated_nodes: 未被支配的节点数
    - is_valid: 是否是合法支配集
    """
    n = graph.num_v
    dominated = [False] * n

    selected_nodes = [u for u, flag in enumerate(selected) if flag]

    for u in selected_nodes:
        dominated[u] = True

    for edge in graph.e[0]:
        u, v = int(edge[0]), int(edge[1])
        if selected[u]:
            dominated[v] = True
        if selected[v]:
            dominated[u] = True

    undominated_nodes = sum(1 for x in dominated if not x)
    is_valid = (undominated_nodes == 0)
    mds_size = len(selected_nodes)

    return mds_size, undominated_nodes, is_valid


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
    selected = random_greedy_mds(adj, seed=seed)
    elapsed = time.time() - start

    mds_size, undominated_nodes, is_valid = evaluate_mds(graph, selected)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "run": run_id,
        "seed": seed,
        "num_nodes": graph.num_v,
        "num_edges": len(graph.e[0]),
        "mds_size": mds_size,
        "undominated_nodes": undominated_nodes,
        "is_valid": is_valid,
        "time_sec": elapsed,
    }

    print("=" * 80)
    print(f"Dataset           : {dataset.name}")
    print(f"Run               : {run_id}")
    print(f"Seed              : {seed}")
    print(f"Path              : {dataset.path}")
    print(f"Nodes             : {graph.num_v}")
    print(f"Edges             : {len(graph.e[0])}")
    print(f"MDS size          : {mds_size}")
    print(f"Undominated nodes : {undominated_nodes}")
    print(f"Valid             : {is_valid}")
    print(f"Time (sec)        : {elapsed:.6f}")
    print("=" * 80)

    return result


def summarize_results(df):
    """
    汇总每个数据集的 Random Greedy 结果：
    - best_mds_size: 多次运行中的最好结果
    - avg_mds_size: 平均支配集大小
    - std_mds_size: 标准差
    """
    summary = []

    for dataset_name, group in df.groupby("dataset_enum"):
        success_group = group[group["mds_size"].notna()].copy()
        valid_group = success_group[success_group["is_valid"] == True].copy()

        if len(success_group) == 0:
            summary.append({
                "dataset_enum": dataset_name,
                "dataset_value": None,
                "num_nodes": None,
                "num_edges": None,
                "best_mds_size": None,
                "best_run": None,
                "best_seed": None,
                "avg_mds_size": None,
                "std_mds_size": None,
                "avg_time_sec": None,
                "std_time_sec": None,
                "valid_runs": 0,
                "success_runs": 0,
            })
            continue

        if len(valid_group) > 0:
            best_row = valid_group.loc[valid_group["mds_size"].idxmin()]
        else:
            best_row = success_group.loc[success_group["mds_size"].idxmin()]

        summary.append({
            "dataset_enum": dataset_name,
            "dataset_value": best_row["dataset_value"],
            "num_nodes": int(success_group["num_nodes"].iloc[0]),
            "num_edges": int(success_group["num_edges"].iloc[0]),
            "best_mds_size": int(best_row["mds_size"]),
            "best_run": int(best_row["run"]),
            "best_seed": int(best_row["seed"]),
            "best_is_valid": bool(best_row["is_valid"]),
            "best_undominated_nodes": int(best_row["undominated_nodes"]),
            "avg_mds_size": float(success_group["mds_size"].mean()),
            "std_mds_size": float(success_group["mds_size"].std(ddof=0)),
            "avg_undominated_nodes": float(success_group["undominated_nodes"].mean()),
            "avg_time_sec": float(success_group["time_sec"].mean()),
            "std_time_sec": float(success_group["time_sec"].std(ddof=0)),
            "valid_runs": int(len(valid_group)),
            "success_runs": int(len(success_group)),
        })

    return pd.DataFrame(summary)


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Random Greedy baseline for MDS")
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
        default="mds_random_greedy_details.csv",
        help="详细结果 csv 文件名",
    )
    parser.add_argument(
        "--summary_name",
        type=str,
        default="mds_random_greedy_summary.csv",
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
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph MDS。")
        datasets = [dataset]

    all_results = []

    for dataset_idx, dataset in enumerate(datasets):
        for run_id in range(1, args.num_runs + 1):
            seed = args.seed + dataset_idx * 1000 + run_id

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
                    "mds_size": None,
                    "undominated_nodes": None,
                    "is_valid": False,
                    "time_sec": None,
                })

            save_dir = ROOT_DIR / "results"
            save_dir.mkdir(parents=True, exist_ok=True)

            partial_path = save_dir / "mds_random_greedy_details_partial.csv"
            pd.DataFrame(all_results).to_csv(
                partial_path,
                index=False,
                encoding="utf-8-sig",
            )

    df_detail = pd.DataFrame(all_results)
    df_summary = summarize_results(df_detail)

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
            "mds_size",
            "undominated_nodes",
            "is_valid",
            "time_sec",
        ]
    ]

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