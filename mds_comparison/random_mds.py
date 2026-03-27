import argparse
import random
import time
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.core import Datasets
from src.utils import from_file_to_graph


def build_adj_from_graph(graph):
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


def evaluate_mds(adj, selected):
    n = len(adj)
    dominated = [False] * n

    for u, flag in enumerate(selected):
        if flag:
            dominated[u] = True
            for v in adj[u]:
                dominated[v] = True

    undominated_nodes = sum(1 for x in dominated if not x)
    is_valid = (undominated_nodes == 0)
    mds_size = sum(1 for x in selected if x)

    return mds_size, undominated_nodes, is_valid


def prune_redundant(selected, adj, seed=42):
    """
    随机顺序删除冗余点：
    若去掉某点后仍然是支配集，就删除它
    """
    rng = random.Random(seed)
    n = len(adj)
    closed = [set([u]) | adj[u] for u in range(n)]

    dom_count = [0] * n
    for u, flag in enumerate(selected):
        if flag:
            for w in closed[u]:
                dom_count[w] += 1

    order = [u for u, flag in enumerate(selected) if flag]
    rng.shuffle(order)

    for u in order:
        if not selected[u]:
            continue
        removable = True
        for w in closed[u]:
            if dom_count[w] <= 1:
                removable = False
                break
        if removable:
            selected[u] = False
            for w in closed[u]:
                dom_count[w] -= 1

    return selected


def random_mds(adj, seed=42):
    """
    随机支配集构造：
    每次从当前未被支配节点 target 出发，
    在 target 的闭邻域中随机选一个点加入解，直到所有点都被支配。
    最后做一次冗余删除。
    """
    rng = random.Random(seed)
    n = len(adj)
    closed = [set([u]) | adj[u] for u in range(n)]

    undominated = set(range(n))
    selected = [False] * n

    while undominated:
        target = rng.choice(tuple(undominated))
        candidates = list(closed[target])
        chosen = rng.choice(candidates)

        if not selected[chosen]:
            selected[chosen] = True
            undominated -= closed[chosen]
        else:
            undominated.discard(target)

    selected = prune_redundant(selected, adj, seed=seed)
    return selected


def solve_one_dataset(dataset, seed=42):
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, _ = build_adj_from_graph(graph)

    start = time.time()
    selected = random_mds(adj, seed=seed)
    elapsed = time.time() - start

    mds_size, undominated_nodes, is_valid = evaluate_mds(adj, selected)

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
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Random baseline for MDS")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="mds_random_results.csv",
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
            result = solve_one_dataset(dataset, seed=args.seed)
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