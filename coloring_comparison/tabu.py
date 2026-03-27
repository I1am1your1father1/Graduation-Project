# coloring_comparison/tabu.py
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


def num_used_colors(colors):
    return max(colors) + 1 if len(colors) > 0 else 0


def greedy_graph_coloring_naive(adj):
    """
    朴素贪心图着色：
    按节点编号顺序依次处理，每次选当前可用的最小颜色
    """
    n = len(adj)
    colors = [-1] * n

    for u in range(n):
        used = set()
        for v in adj[u]:
            if colors[v] != -1:
                used.add(colors[v])

        c = 0
        while c in used:
            c += 1
        colors[u] = c

    return colors


def random_greedy_coloring(adj, seed=42):
    """
    随机顺序贪心着色：
    打乱节点顺序后，仍然使用“最小可用颜色”
    """
    rng = random.Random(seed)
    n = len(adj)
    order = list(range(n))
    rng.shuffle(order)

    colors = [-1] * n

    for u in order:
        used = set()
        for v in adj[u]:
            if colors[v] != -1:
                used.add(colors[v])

        c = 0
        while c in used:
            c += 1
        colors[u] = c

    return colors


def evaluate_coloring(graph, colors):
    """
    评估着色结果：
    - num_colors: 实际使用颜色数
    - conflict_edges: 冲突边数
    - is_valid: 是否合法
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


def color_class_sizes(colors, k):
    sizes = [0] * k
    for c in colors:
        sizes[c] += 1
    return sizes


def build_infeasible_k_coloring_from_legal(base_colors, target_k, remove_color, seed=42):
    """
    从一个合法的 (target_k + 1) 着色出发，删除一种颜色类，
    把该颜色类中的点随机塞回 [0, target_k-1]，得到一个可能冲突的 target_k 着色
    """
    rng = random.Random(seed)
    n = len(base_colors)
    new_colors = [-1] * n

    for v in range(n):
        c = base_colors[v]
        if c < remove_color:
            new_colors[v] = c
        elif c > remove_color:
            new_colors[v] = c - 1

    for v in range(n):
        if base_colors[v] == remove_color:
            new_colors[v] = rng.randrange(target_k)

    return new_colors


def initialize_conflict_state(adj, colors):
    """
    初始化：
    - same_color_deg[v]: 与 v 同色的邻居个数
    - total_conflict_edges: 冲突边总数
    - conflict_vertices: 当前有冲突的点集合
    """
    n = len(adj)
    same_color_deg = [0] * n
    total_conflict_edges = 0

    for u in range(n):
        for v in adj[u]:
            if u < v and colors[u] == colors[v]:
                same_color_deg[u] += 1
                same_color_deg[v] += 1
                total_conflict_edges += 1

    conflict_vertices = set(u for u in range(n) if same_color_deg[u] > 0)
    return same_color_deg, total_conflict_edges, conflict_vertices


def get_neighbor_color_count(v, adj, colors):
    """
    统计 v 的邻居颜色分布
    """
    cnt = {}
    for u in adj[v]:
        c = colors[u]
        cnt[c] = cnt.get(c, 0) + 1
    return cnt


def pick_one_absent_color(neighbor_color_count, old_color, k, rng):
    """
    在 [0, k-1] 中找一个没有出现在邻居颜色里的颜色
    """
    if len(neighbor_color_count) >= k:
        return None

    start = rng.randrange(k)
    for offset in range(k):
        c = (start + offset) % k
        if c != old_color and c not in neighbor_color_count:
            return c
    return None


def tabu_feasibility_search(
    adj,
    init_colors,
    k,
    max_steps=2000,
    tabu_tenure=7,
    seed=42,
):
    """
    在固定颜色数 k 下做 tabu feasibility search，
    目标是把 conflict_edges 降到 0
    """
    rng = random.Random(seed)
    n = len(adj)

    colors = init_colors.copy()
    same_color_deg, total_conflict_edges, conflict_vertices = initialize_conflict_state(adj, colors)

    best_colors = colors.copy()
    best_conflict_edges = total_conflict_edges

    tabu_until = {}

    for step in range(1, max_steps + 1):
        if total_conflict_edges == 0:
            return colors, True

        best_move = None
        best_new_conflicts = None

        current_conflict_vertices = list(conflict_vertices)
        rng.shuffle(current_conflict_vertices)

        for v in current_conflict_vertices:
            old_color = colors[v]
            neighbor_color_count = get_neighbor_color_count(v, adj, colors)
            old_same = neighbor_color_count.get(old_color, 0)

            candidate_colors = []

            for c in neighbor_color_count.keys():
                if c != old_color:
                    candidate_colors.append(c)

            absent_color = pick_one_absent_color(neighbor_color_count, old_color, k, rng)
            if absent_color is not None:
                candidate_colors.append(absent_color)

            candidate_colors = list(set(candidate_colors))

            for new_color in candidate_colors:
                if new_color == old_color:
                    continue

                new_same = neighbor_color_count.get(new_color, 0)
                new_total_conflicts = total_conflict_edges + (new_same - old_same)

                is_tabu = step < tabu_until.get((v, new_color), 0)

                if is_tabu and new_total_conflicts >= best_conflict_edges:
                    continue

                if best_move is None or new_total_conflicts < best_new_conflicts:
                    best_move = (v, old_color, new_color, old_same, new_same)
                    best_new_conflicts = new_total_conflicts
                elif new_total_conflicts == best_new_conflicts:
                    if rng.random() < 0.5:
                        best_move = (v, old_color, new_color, old_same, new_same)
                        best_new_conflicts = new_total_conflicts

        if best_move is None:
            if not conflict_vertices:
                break

            v = rng.choice(list(conflict_vertices))
            old_color = colors[v]
            new_color = rng.randrange(k)
            while new_color == old_color:
                new_color = rng.randrange(k)

            neighbor_color_count = get_neighbor_color_count(v, adj, colors)
            old_same = neighbor_color_count.get(old_color, 0)
            new_same = neighbor_color_count.get(new_color, 0)
            best_move = (v, old_color, new_color, old_same, new_same)
            best_new_conflicts = total_conflict_edges + (new_same - old_same)

        v, old_color, new_color, old_same, new_same = best_move

        colors[v] = new_color
        total_conflict_edges = best_new_conflicts

        for u in adj[v]:
            if colors[u] == old_color:
                same_color_deg[u] -= 1
                if same_color_deg[u] > 0:
                    conflict_vertices.add(u)
                else:
                    conflict_vertices.discard(u)

            if colors[u] == new_color:
                same_color_deg[u] += 1
                if same_color_deg[u] > 0:
                    conflict_vertices.add(u)
                else:
                    conflict_vertices.discard(u)

        same_color_deg[v] = new_same
        if same_color_deg[v] > 0:
            conflict_vertices.add(v)
        else:
            conflict_vertices.discard(v)

        tabu_until[(v, old_color)] = step + tabu_tenure + rng.randint(0, 2)

        if total_conflict_edges < best_conflict_edges:
            best_conflict_edges = total_conflict_edges
            best_colors = colors.copy()

    return best_colors, (best_conflict_edges == 0)


def tabu_graph_coloring(
    adj,
    init_mode="greedy",
    max_steps_per_k=2000,
    tabu_tenure=7,
    restarts=3,
    seed=42,
):
    """
    Graph Coloring 的 Tabu Search 主过程：
    - 先得到一个合法初始着色
    - 然后逐步尝试把颜色数从 k 降到 k-1
    - 每次降色时用 tabu feasibility search 尝试消除冲突
    """
    rng = random.Random(seed)

    if init_mode == "greedy":
        best_legal_colors = greedy_graph_coloring_naive(adj)
    elif init_mode == "random":
        best_legal_colors = random_greedy_coloring(adj, seed=seed)
    else:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    best_k = num_used_colors(best_legal_colors)

    while best_k > 1:
        target_k = best_k - 1
        sizes = color_class_sizes(best_legal_colors, best_k)

        color_order = sorted(range(best_k), key=lambda c: (sizes[c], c))

        success = False

        num_trials = min(restarts, len(color_order))
        for trial in range(num_trials):
            remove_color = color_order[trial]

            init_infeasible = build_infeasible_k_coloring_from_legal(
                best_legal_colors,
                target_k=target_k,
                remove_color=remove_color,
                seed=seed + trial,
            )

            candidate_colors, feasible = tabu_feasibility_search(
                adj=adj,
                init_colors=init_infeasible,
                k=target_k,
                max_steps=max_steps_per_k,
                tabu_tenure=tabu_tenure,
                seed=seed + trial,
            )

            if feasible:
                best_legal_colors = candidate_colors
                best_k = target_k
                success = True
                break

        if not success:
            break

    return best_legal_colors


def solve_one_dataset(dataset, init_mode="greedy", max_steps_per_k=2000, tabu_tenure=7, restarts=3, seed=42):
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
    colors = tabu_graph_coloring(
        adj=adj,
        init_mode=init_mode,
        max_steps_per_k=max_steps_per_k,
        tabu_tenure=tabu_tenure,
        restarts=restarts,
        seed=seed,
    )
    elapsed = time.time() - start

    num_colors, conflict_edges, is_valid = evaluate_coloring(graph, colors)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(graph.e[0]),
        "num_colors": num_colors,
        "conflict_edges": conflict_edges,
        "is_valid": is_valid,
        "time_sec": elapsed,
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(graph.e[0])}")
    print(f"Num colors      : {num_colors}")
    print(f"Conflict edges  : {conflict_edges}")
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
    parser = argparse.ArgumentParser(description="Tabu Search baseline for Graph Coloring")
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
        help="初始解方式",
    )
    parser.add_argument(
        "--max_steps_per_k",
        type=int,
        default=2000,
        help="每个 k 下 tabu feasibility search 的最大步数",
    )
    parser.add_argument(
        "--tabu_tenure",
        type=int,
        default=7,
        help="tabu tenure 长度",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="每个 k 的重启次数",
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
        default="coloring_tabu_results.csv",
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
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph coloring。")
        datasets = [dataset]

    all_results = []

    for dataset in datasets:
        try:
            result = solve_one_dataset(
                dataset=dataset,
                init_mode=args.init_mode,
                max_steps_per_k=args.max_steps_per_k,
                tabu_tenure=args.tabu_tenure,
                restarts=args.restarts,
                seed=args.seed,
            )
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
                "num_colors": None,
                "conflict_edges": None,
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
            "num_colors",
            "conflict_edges",
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