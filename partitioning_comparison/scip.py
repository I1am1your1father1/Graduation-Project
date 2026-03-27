# partitioning_comparison/scip_solver.py
# -*- coding: utf-8 -*-

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


def greedy_balanced_partition(adj, num_parts=2):
    """
    用一个简单 greedy 先给 SCIP 一个初始可行解和 fallback 解
    目标：尽量均衡，同时让新增 cut 边尽量少
    """
    n = len(adj)
    if num_parts <= 0:
        raise ValueError("num_parts 必须为正整数。")
    if num_parts > n:
        raise ValueError("num_parts 不能大于节点数。")

    part = [-1] * n
    degree = [len(adj[u]) for u in range(n)]

    base = n // num_parts
    extra = n % num_parts
    capacities = [base + (1 if i < extra else 0) for i in range(num_parts)]
    part_sizes = [0] * num_parts

    order = sorted(range(n), key=lambda u: (-degree[u], u))

    for u in order:
        best_p = None
        best_inc_cut = None
        best_size = None

        for p in range(num_parts):
            if part_sizes[p] >= capacities[p]:
                continue

            inc_cut = 0
            for v in adj[u]:
                if part[v] == -1:
                    continue
                if part[v] != p:
                    inc_cut += 1

            if best_p is None:
                best_p = p
                best_inc_cut = inc_cut
                best_size = part_sizes[p]
            else:
                if inc_cut < best_inc_cut:
                    best_p = p
                    best_inc_cut = inc_cut
                    best_size = part_sizes[p]
                elif inc_cut == best_inc_cut:
                    if part_sizes[p] < best_size:
                        best_p = p
                        best_inc_cut = inc_cut
                        best_size = part_sizes[p]
                    elif part_sizes[p] == best_size and p < best_p:
                        best_p = p
                        best_inc_cut = inc_cut
                        best_size = part_sizes[p]

        if best_p is None:
            raise RuntimeError("没有可用分区可分配，检查容量逻辑。")

        part[u] = best_p
        part_sizes[best_p] += 1

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


def solve_partition_with_scip(num_nodes, edges, num_parts, fallback_part, time_limit=3600.0, verbose=False):
    """
    用 SCIP 求 balanced k-way graph partitioning

    变量：
        x[u,p] in {0,1}  点 u 是否分到分区 p
        y[e]   in {0,1}  边 e 是否跨分区

    目标：
        min sum_e y[e]

    约束：
        1) 每个点恰好属于一个分区
        2) 每个分区大小精确受 floor/ceil 容量约束
        3) 若边两端被分到不同分区，则 y[e] 必须为 1
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError as e:
        raise ImportError("未安装 pyscipopt，无法使用 SCIP。") from e

    if num_parts <= 0:
        raise ValueError("num_parts 必须为正整数。")
    if num_parts > num_nodes:
        raise ValueError("num_parts 不能大于节点数。")

    model = Model("GraphPartitioning")

    if not verbose:
        model.hideOutput()

    model.setRealParam("limits/time", float(time_limit))

    x = {}
    y = {}

    for u in range(num_nodes):
        for p in range(num_parts):
            x[(u, p)] = model.addVar(vtype="B", name=f"x_{u}_{p}")

    for idx, (u, v) in enumerate(edges):
        y[idx] = model.addVar(vtype="B", name=f"y_{u}_{v}")

    for u in range(num_nodes):
        model.addCons(quicksum(x[(u, p)] for p in range(num_parts)) == 1)

    base = num_nodes // num_parts
    extra = num_nodes % num_parts
    capacities = [base + (1 if p < extra else 0) for p in range(num_parts)]

    for p in range(num_parts):
        model.addCons(quicksum(x[(u, p)] for u in range(num_nodes)) == capacities[p])

    for idx, (u, v) in enumerate(edges):
        for p in range(num_parts):
            model.addCons(y[idx] >= x[(u, p)] - x[(v, p)])
            model.addCons(y[idx] >= x[(v, p)] - x[(u, p)])

    for p in range(min(num_parts, num_nodes)):
        model.addCons(x[(p, p)] == 1)

    model.setObjective(quicksum(y[idx] for idx in range(len(edges))), "minimize")

    try:
        sol = model.createSol()
        for u in range(num_nodes):
            assigned_p = fallback_part[u]
            for p in range(num_parts):
                model.setSolVal(sol, x[(u, p)], 1.0 if p == assigned_p else 0.0)

        for idx, (u, v) in enumerate(edges):
            model.setSolVal(sol, y[idx], 1.0 if fallback_part[u] != fallback_part[v] else 0.0)

        model.addSol(sol, free=True)
    except Exception:
        pass

    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    sol = model.getBestSol()

    if sol is None:
        return {
            "part": fallback_part,
            "solve_time_sec": elapsed,
        }

    part = [-1] * num_nodes
    for u in range(num_nodes):
        assigned = False
        for p in range(num_parts):
            if model.getSolVal(sol, x[(u, p)]) > 0.5:
                part[u] = p
                assigned = True
                break
        if not assigned:
            part[u] = fallback_part[u]

    return {
        "part": part,
        "solve_time_sec": elapsed,
    }


def solve_one_dataset(dataset, num_parts=2, time_limit=3600.0, verbose=False):
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

    fallback_part = greedy_balanced_partition(
        adj=adj,
        num_parts=num_parts,
    )

    solve_result = solve_partition_with_scip(
        num_nodes=graph.num_v,
        edges=edges,
        num_parts=num_parts,
        fallback_part=fallback_part,
        time_limit=time_limit,
        verbose=verbose,
    )

    part = solve_result["part"]

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
        "time_sec": solve_result["solve_time_sec"],
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
    print(f"Time (sec)      : {solve_result['solve_time_sec']:.6f}")
    print("=" * 90)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="SCIP exact solver for Graph Partitioning")
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
        "--time_limit",
        type=float,
        default=3600.0,
        help="每个图的时间限制（秒），默认 3600 秒",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否显示 SCIP 日志",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="partition_scip_results.csv",
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
                time_limit=args.time_limit,
                verbose=args.verbose,
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