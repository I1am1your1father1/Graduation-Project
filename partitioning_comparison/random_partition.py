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


def build_edge_list_from_graph(graph):
    """
    从图中提取无向边列表，每条边只保留一次
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


def random_balanced_partition(num_nodes, num_parts, seed=42):
    """
    随机均衡划分：
    将 n 个点尽量均匀地随机分到 num_parts 个分区中。
    返回 part，其中 part[u] 表示点 u 属于哪个分区。
    """
    if num_parts <= 0:
        raise ValueError("num_parts 必须为正整数。")
    if num_parts > num_nodes:
        raise ValueError("num_parts 不能大于节点数。")

    rng = random.Random(seed)
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)

    base = num_nodes // num_parts
    extra = num_nodes % num_parts

    capacities = [base + (1 if i < extra else 0) for i in range(num_parts)]

    part = [-1] * num_nodes
    idx = 0
    for p in range(num_parts):
        for _ in range(capacities[p]):
            u = nodes[idx]
            part[u] = p
            idx += 1

    return part


def evaluate_partition(num_nodes, edges, part, num_parts):
    """
    评估图划分结果：
    - cut_edges: 跨分区边数（越小越好）
    - part_sizes: 每个分区的大小
    - balance_gap: 最大分区与最小分区的大小差
    - is_valid: 是否所有点都被分配到了合法分区
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


def solve_one_dataset(dataset, num_parts=2, seed=42):
    """
    求解单个数据集
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    edges = build_edge_list_from_graph(graph)

    start = time.time()
    part = random_balanced_partition(
        num_nodes=graph.num_v,
        num_parts=num_parts,
        seed=seed,
    )
    elapsed = time.time() - start

    cut_edges, part_sizes, balance_gap, is_valid = evaluate_partition(
        num_nodes=graph.num_v,
        edges=edges,
        part=part,
        num_parts=num_parts,
    )

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(edges),
        "num_parts": num_parts,
        "cut_edges": cut_edges,
        "balance_gap": balance_gap,
        "is_valid": is_valid,
        "time_sec": elapsed,
        "part_sizes": " ".join(map(str, part_sizes)) if part_sizes is not None else "",
    }

    print("=" * 90)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(edges)}")
    print(f"Num parts       : {num_parts}")
    print(f"Cut edges       : {cut_edges}")
    print(f"Part sizes      : {part_sizes}")
    print(f"Balance gap     : {balance_gap}")
    print(f"Valid           : {is_valid}")
    print(f"Time (sec)      : {elapsed:.6f}")
    print("=" * 90)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Random balanced partition baseline for Graph Partitioning")
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
        help="分区数 k，默认 2",
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
        default="partition_random_results.csv",
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
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph partitioning。")
        datasets = [dataset]

    all_results = []

    for dataset in datasets:
        try:
            result = solve_one_dataset(
                dataset=dataset,
                num_parts=args.num_parts,
                seed=args.seed,
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
                "cut_edges": None,
                "balance_gap": None,
                "is_valid": False,
                "time_sec": None,
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
            "cut_edges",
            "balance_gap",
            "is_valid",
            "time_sec",
            "part_sizes",
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