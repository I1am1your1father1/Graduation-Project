# partitioning_comparison/KL.py
# -*- coding: utf-8 -*-

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


def random_balanced_bipartition(num_nodes, seed=42):
    """
    生成一个尽量均衡的二分初始解
    """
    rng = random.Random(seed)
    nodes = list(range(num_nodes))
    rng.shuffle(nodes)

    size_a = (num_nodes + 1) // 2
    part = [0] * num_nodes
    for i, u in enumerate(nodes):
        part[u] = 0 if i < size_a else 1
    return part


def greedy_balanced_bipartition(adj):
    """
    一个简单的 balanced greedy 二分初始解
    """
    n = len(adj)
    degree = [len(adj[u]) for u in range(n)]
    order = sorted(range(n), key=lambda u: (-degree[u], u))

    cap0 = (n + 1) // 2
    cap1 = n // 2
    size0 = 0
    size1 = 0

    part = [-1] * n

    for u in order:
        inc0 = 0
        inc1 = 0
        for v in adj[u]:
            if part[v] == -1:
                continue
            if part[v] != 0:
                inc0 += 1
            if part[v] != 1:
                inc1 += 1

        candidates = []
        if size0 < cap0:
            candidates.append((inc0, size0, 0))
        if size1 < cap1:
            candidates.append((inc1, size1, 1))

        if not candidates:
            raise RuntimeError("没有可用分区可分配，检查容量逻辑。")

        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        chosen = candidates[0][2]
        part[u] = chosen
        if chosen == 0:
            size0 += 1
        else:
            size1 += 1

    return part


def evaluate_partition(num_nodes, edges, part, num_parts=2):
    """
    评估图划分结果：
    - cut_edges: 跨分区边数（越大说明切开的边更多）
    - part_sizes: 每个分区大小
    - balance_gap: 最大分区与最小分区大小差
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


def compute_d_values(adj, part):
    """
    对当前二分划分计算每个点的 D 值：
    D(v) = external(v) - internal(v)
    """
    n = len(adj)
    D = [0] * n
    for u in range(n):
        ext_u = 0
        int_u = 0
        pu = part[u]
        for v in adj[u]:
            if part[v] == pu:
                int_u += 1
            else:
                ext_u += 1
        D[u] = ext_u - int_u
    return D


def kl_single_pass(adj, part, start_time=None, time_limit=None):
    """
    执行一轮 Kernighan-Lin pass：
    - 锁定点
    - 反复选择一个跨分区交换对 (a,b)
    - 记录 gain 序列
    - 应用 gain 前缀和最大的前缀（若最大前缀和 > 0）

    返回：
    - improved: 本轮是否成功改进
    - new_part: 改进后的划分
    """
    n = len(adj)
    new_part = part.copy()

    A = [u for u in range(n) if new_part[u] == 0]
    B = [u for u in range(n) if new_part[u] == 1]

    if len(A) != len(B) and len(A) != len(B) + 1 and len(B) != len(A) + 1:
        raise ValueError("KL 需要初始划分基本均衡。")

    locked = [False] * n
    swap_pairs = []
    gains = []

    D = compute_d_values(adj, new_part)

    num_pairs = min(len(A), len(B))
    for _ in range(num_pairs):
        if start_time is not None and time_limit is not None:
            if time.time() - start_time >= time_limit:
                return False, part

        best_a = None
        best_b = None
        best_gain = None

        for a in A:
            if locked[a]:
                continue
            for b in B:
                if locked[b]:
                    continue

                if start_time is not None and time_limit is not None:
                    if time.time() - start_time >= time_limit:
                        return False, part

                cab = 1 if b in adj[a] else 0
                gain = D[a] + D[b] - 2 * cab

                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_a = a
                    best_b = b

        if best_a is None or best_b is None:
            break

        swap_pairs.append((best_a, best_b))
        gains.append(best_gain)

        locked[best_a] = True
        locked[best_b] = True

        # 在“临时交换”的意义下更新 D 值，供后续选点
        for v in range(n):
            if locked[v]:
                continue

            if new_part[v] == 0:
                delta = 0
                if best_a in adj[v]:
                    delta += 2
                if best_b in adj[v]:
                    delta -= 2
                D[v] += delta
            else:
                delta = 0
                if best_a in adj[v]:
                    delta -= 2
                if best_b in adj[v]:
                    delta += 2
                D[v] += delta

    if not gains:
        return False, part

    best_prefix_sum = None
    current_sum = 0
    best_k = -1
    for i, g in enumerate(gains):
        current_sum += g
        if best_prefix_sum is None or current_sum > best_prefix_sum:
            best_prefix_sum = current_sum
            best_k = i

    if best_prefix_sum is None or best_prefix_sum <= 0:
        return False, part

    final_part = part.copy()
    for i in range(best_k + 1):
        a, b = swap_pairs[i]
        final_part[a] = 1
        final_part[b] = 0

    return True, final_part


def kernighan_lin_bipartition(adj, init_mode="greedy", max_passes=20, seed=42, time_limit=600.0):
    """
    KL 二分划分主过程
    注意：KL 是经典的二分算法，这里只支持 2-way partition
    """
    n = len(adj)

    if init_mode == "greedy":
        part = greedy_balanced_bipartition(adj)
    elif init_mode == "random":
        part = random_balanced_bipartition(n, seed=seed)
    else:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    start_time = time.time()

    for _ in range(max_passes):
        if time.time() - start_time >= time_limit:
            break

        improved, new_part = kl_single_pass(
            adj,
            part,
            start_time=start_time,
            time_limit=time_limit,
        )
        if not improved:
            break
        part = new_part

    return part


def solve_one_dataset(dataset, init_mode="greedy", max_passes=20, seed=42, time_limit=600.0):
    """
    求解单个数据集
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, _ = build_adj_from_graph(graph)
    edges = build_edge_list_from_graph(graph)

    start = time.time()
    part = kernighan_lin_bipartition(
        adj=adj,
        init_mode=init_mode,
        max_passes=max_passes,
        seed=seed,
        time_limit=time_limit,
    )
    elapsed = time.time() - start

    cut_edges, part_sizes, balance_gap, is_valid = evaluate_partition(
        num_nodes=graph.num_v,
        edges=edges,
        part=part,
        num_parts=2,
    )

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(edges),
        "num_parts": 2,
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
    print(f"Num parts       : 2")
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
    parser = argparse.ArgumentParser(description="Kernighan-Lin baseline for Graph Partitioning (2-way only)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        default="greedy",
        choices=["greedy", "random"],
        help="初始划分方式",
    )
    parser.add_argument(
        "--max_passes",
        type=int,
        default=20,
        help="KL 最大 pass 数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（仅 random 初始化时主要生效）",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=1200.0,
        help="KL 时间限制（秒），默认 600 秒",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="partition_kl_results.csv",
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
                init_mode=args.init_mode,
                max_passes=args.max_passes,
                seed=args.seed,
                time_limit=args.time_limit,
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
                "num_parts": 2,
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