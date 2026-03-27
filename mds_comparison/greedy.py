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


def greedy_mds_naive(adj):
    """
    朴素 greedy 求最小支配集：
    每一步选择能新支配最多“当前未被支配节点”的点
    """
    n = len(adj)

    # closed_neighborhood[u] = {u} U N(u)
    closed_neighborhood = [set([u]) | adj[u] for u in range(n)]

    undominated = set(range(n))
    selected = [False] * n

    while undominated:
        best_u = None
        best_gain = None
        best_deg = None

        for u in range(n):
            gain = len(closed_neighborhood[u] & undominated)
            deg_u = len(adj[u])

            if best_u is None:
                best_u = u
                best_gain = gain
                best_deg = deg_u
            else:
                if gain > best_gain:
                    best_u = u
                    best_gain = gain
                    best_deg = deg_u
                elif gain == best_gain:
                    if deg_u > best_deg:
                        best_u = u
                        best_gain = gain
                        best_deg = deg_u
                    elif deg_u == best_deg and u < best_u:
                        best_u = u
                        best_gain = gain
                        best_deg = deg_u

        if best_u is None or best_gain == 0:
            u = min(undominated)
            best_u = u

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
        for v in graph.e[0]:
            pass

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
    selected = greedy_mds_naive(adj)
    elapsed = time.time() - start

    mds_size, undominated_nodes, is_valid = evaluate_mds(graph, selected)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(graph.e[0]),
        "mds_size": mds_size,
        "undominated_nodes": undominated_nodes,
        "is_valid": is_valid,
        "time_sec": elapsed,
    }

    print("=" * 80)
    print(f"Dataset           : {dataset.name}")
    print(f"Path              : {dataset.path}")
    print(f"Nodes             : {graph.num_v}")
    print(f"Edges             : {len(graph.e[0])}")
    print(f"MDS size          : {mds_size}")
    print(f"Undominated nodes : {undominated_nodes}")
    print(f"Valid             : {is_valid}")
    print(f"Time (sec)        : {elapsed:.6f}")
    print("=" * 80)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Naive Greedy baseline for MDS")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="mds_greedy_results.csv",
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
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph MDS。")
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
                "mds_size": None,
                "undominated_nodes": None,
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
            "mds_size",
            "undominated_nodes",
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