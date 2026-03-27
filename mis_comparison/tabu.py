import argparse
import random
import time
from collections import defaultdict
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


def count_selected(in_set):
    return sum(1 for x in in_set if x)


def greedy_mis_min_degree_naive(adj):
    """
    朴素最小度贪心 MIS，作为 tabu 的一种初始解
    """
    n = len(adj)
    alive = set(range(n))
    in_set = [False] * n

    while alive:
        best_u = None
        best_deg = None

        for u in sorted(alive):
            deg_u = sum((v in alive) for v in adj[u])
            if best_deg is None or deg_u < best_deg:
                best_deg = deg_u
                best_u = u

        in_set[best_u] = True

        to_remove = {best_u}
        for v in adj[best_u]:
            if v in alive:
                to_remove.add(v)

        alive -= to_remove

    return in_set


def random_maximal_independent_set(adj, seed=42):
    """
    随机生成一个 maximal independent set
    """
    rng = random.Random(seed)
    n = len(adj)
    order = list(range(n))
    rng.shuffle(order)

    in_set = [False] * n
    blocked = [False] * n

    for u in order:
        if not blocked[u]:
            in_set[u] = True
            blocked[u] = True
            for v in adj[u]:
                blocked[v] = True

    return in_set


def make_maximal(in_set, adj, degree, blocked_add=None, rng=None, randomized=False):
    """
    将当前独立集补成 maximal independent set
    blocked_add: 本轮补全过程中不允许重新加入的点（用于避免立即回退）
    """
    if blocked_add is None:
        blocked_add = set()

    n = len(adj)
    candidates = [u for u in range(n) if (not in_set[u]) and (u not in blocked_add)]

    if randomized and rng is not None:
        candidates.sort(key=lambda u: (degree[u], rng.random()))
    else:
        candidates.sort(key=lambda u: (degree[u], u))

    for u in candidates:
        if in_set[u] or u in blocked_add:
            continue

        can_add = True
        for v in adj[u]:
            if in_set[v]:
                can_add = False
                break

        if can_add:
            in_set[u] = True

    return in_set


def build_selected_conflict_info(in_set, adj):
    """
    对每个未选点，统计它与当前独立集中多少个点冲突
    如果恰好只和一个已选点冲突，则记录这个唯一冲突点
    """
    n = len(adj)
    selected_neighbor_count = [0] * n
    unique_selected_neighbor = [-1] * n

    for u, sel in enumerate(in_set):
        if not sel:
            continue
        for v in adj[u]:
            if in_set[v]:
                continue
            selected_neighbor_count[v] += 1
            if selected_neighbor_count[v] == 1:
                unique_selected_neighbor[v] = u
            else:
                unique_selected_neighbor[v] = -2

    return selected_neighbor_count, unique_selected_neighbor


def apply_move_and_complete(current_in_set, adj, degree, removed_vertices, added_vertices, rng, randomized_completion=False):
    """
    应用 move 后再补成 maximal independent set
    """
    new_in_set = current_in_set.copy()

    for u in removed_vertices:
        new_in_set[u] = False
    for v in added_vertices:
        new_in_set[v] = True

    blocked_add = set(removed_vertices)
    new_in_set = make_maximal(
        new_in_set,
        adj,
        degree,
        blocked_add=blocked_add,
        rng=rng,
        randomized=randomized_completion,
    )
    return new_in_set


def diff_vertices(old_in_set, new_in_set):
    removed = []
    added = []
    for i in range(len(old_in_set)):
        if old_in_set[i] and not new_in_set[i]:
            removed.append(i)
        elif (not old_in_set[i]) and new_in_set[i]:
            added.append(i)
    return removed, added


def find_best_12_exchange_move(
    in_set,
    adj,
    degree,
    tabu_add_until,
    tabu_remove_until,
    step,
    best_size,
):
    """
    寻找一个 (1,2)-exchange 改进：
    删除一个已选点 u，加入两个未选点 a,b
    要求：
      1) a,b 都只和当前解中的 u 冲突
      2) a,b 彼此不相邻
    """
    current_size = count_selected(in_set)
    aspiration = (current_size + 1 > best_size)

    selected_neighbor_count, unique_selected_neighbor = build_selected_conflict_info(in_set, adj)

    owner_to_candidates = defaultdict(list)
    n = len(adj)

    for v in range(n):
        if in_set[v]:
            continue
        if selected_neighbor_count[v] == 1 and unique_selected_neighbor[v] >= 0:
            owner = unique_selected_neighbor[v]
            owner_to_candidates[owner].append(v)

    selected_vertices = [u for u, sel in enumerate(in_set) if sel]
    selected_vertices.sort(key=lambda u: (-len(owner_to_candidates[u]), degree[u], u))

    for u in selected_vertices:
        if step < tabu_remove_until[u] and not aspiration:
            continue

        cands = []
        for v in owner_to_candidates.get(u, []):
            if step < tabu_add_until[v] and not aspiration:
                continue
            cands.append(v)

        if len(cands) < 2:
            continue

        cands.sort(key=lambda x: (degree[x], x))

        for i in range(len(cands)):
            a = cands[i]
            for j in range(i + 1, len(cands)):
                b = cands[j]
                if b not in adj[a]:
                    return ([u], [a, b])

    return None


def find_best_swap_move(
    in_set,
    adj,
    degree,
    tabu_add_until,
    tabu_remove_until,
    step,
):
    """
    寻找一个 (1,1)-swap：
    删除一个已选点 u，加入一个未选点 v
    要求 v 只和当前解中的 u 冲突
    """
    selected_neighbor_count, unique_selected_neighbor = build_selected_conflict_info(in_set, adj)

    owner_to_candidates = defaultdict(list)
    n = len(adj)

    for v in range(n):
        if in_set[v]:
            continue
        if selected_neighbor_count[v] == 1 and unique_selected_neighbor[v] >= 0:
            owner = unique_selected_neighbor[v]
            owner_to_candidates[owner].append(v)

    selected_vertices = [u for u, sel in enumerate(in_set) if sel]
    selected_vertices.sort(key=lambda u: (-len(owner_to_candidates[u]), degree[u], u))

    for u in selected_vertices:
        if step < tabu_remove_until[u]:
            continue

        cands = []
        for v in owner_to_candidates.get(u, []):
            if step < tabu_add_until[v]:
                continue
            cands.append(v)

        if not cands:
            continue

        cands.sort(key=lambda x: (degree[x], x))
        v = cands[0]
        return ([u], [v])

    return None


def perturbation_move(in_set, adj, degree, tabu_remove_until, step, rng):
    """
    当没有可用交换时，做一次扰动：
    删掉一个已选点，再随机补成 maximal independent set
    """
    selected = [u for u, sel in enumerate(in_set) if sel and step >= tabu_remove_until[u]]
    if not selected:
        selected = [u for u, sel in enumerate(in_set) if sel]

    if not selected:
        return in_set.copy()

    selected.sort(key=lambda u: (-degree[u], u))
    top_k = selected[:min(5, len(selected))]
    u = rng.choice(top_k)

    new_in_set = in_set.copy()
    new_in_set[u] = False
    new_in_set = make_maximal(
        new_in_set,
        adj,
        degree,
        blocked_add={u},
        rng=rng,
        randomized=True,
    )
    return new_in_set


def tabu_search_mis(
    adj,
    init_mode="greedy",
    max_steps=300,
    tabu_tenure=7,
    seed=42,
):
    """
    MIS 的 tabu search 主过程
    """
    rng = random.Random(seed)
    degree = [len(nei) for nei in adj]
    n = len(adj)

    if init_mode == "greedy":
        current = greedy_mis_min_degree_naive(adj)
    elif init_mode == "random":
        current = random_maximal_independent_set(adj, seed=seed)
    else:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    current = make_maximal(current, adj, degree, rng=rng, randomized=False)

    best = current.copy()
    best_size = count_selected(best)

    tabu_add_until = [0] * n
    tabu_remove_until = [0] * n

    for step in range(1, max_steps + 1):
        move = find_best_12_exchange_move(
            current,
            adj,
            degree,
            tabu_add_until,
            tabu_remove_until,
            step,
            best_size,
        )

        if move is not None:
            removed_vertices, added_vertices = move
            new_current = apply_move_and_complete(
                current,
                adj,
                degree,
                removed_vertices,
                added_vertices,
                rng=rng,
                randomized_completion=False,
            )
        else:
            move = find_best_swap_move(
                current,
                adj,
                degree,
                tabu_add_until,
                tabu_remove_until,
                step,
            )

            if move is not None:
                removed_vertices, added_vertices = move
                new_current = apply_move_and_complete(
                    current,
                    adj,
                    degree,
                    removed_vertices,
                    added_vertices,
                    rng=rng,
                    randomized_completion=True,
                )
            else:
                new_current = perturbation_move(
                    current,
                    adj,
                    degree,
                    tabu_remove_until,
                    step,
                    rng,
                )

        removed_vertices, added_vertices = diff_vertices(current, new_current)

        for u in removed_vertices:
            tabu_add_until[u] = step + tabu_tenure
        for v in added_vertices:
            tabu_remove_until[v] = step + tabu_tenure

        current = new_current
        current_size = count_selected(current)

        if current_size > best_size:
            best = current.copy()
            best_size = current_size

    return best


def check_mis_valid_and_violations(graph, mis_nodes):
    """
    检查解是否合法，并统计冲突边数
    """
    mis_set = set(mis_nodes)
    violation_edges = 0

    for edge in graph.e[0]:
        u, v = int(edge[0]), int(edge[1])
        if u in mis_set and v in mis_set:
            violation_edges += 1

    return violation_edges == 0, violation_edges


def solve_one_dataset(dataset, init_mode="greedy", max_steps=300, tabu_tenure=7, seed=42):
    """
    求解单个数据集，并输出与 greedy 相同格式的结果
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, _ = build_adj_from_graph(graph)

    start = time.time()
    in_set = tabu_search_mis(
        adj=adj,
        init_mode=init_mode,
        max_steps=max_steps,
        tabu_tenure=tabu_tenure,
        seed=seed,
    )
    elapsed = time.time() - start

    mis_nodes = [i for i, sel in enumerate(in_set) if sel]
    mis_size = len(mis_nodes)
    is_valid, violation_edges = check_mis_valid_and_violations(graph, mis_nodes)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(graph.e[0]),
        "mis_size": mis_size,
        "violation_edges": violation_edges,
        "is_valid": is_valid,
        "time_sec": elapsed,
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(graph.e[0])}")
    print(f"MIS size        : {mis_size}")
    print(f"Violation edges : {violation_edges}")
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
    parser = argparse.ArgumentParser(description="Tabu Search baseline for MIS")
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
        "--max_steps",
        type=int,
        default=300,
        help="tabu search 最大迭代步数",
    )
    parser.add_argument(
        "--tabu_tenure",
        type=int,
        default=7,
        help="tabu tenure 长度",
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
        default="mis_tabu_results.csv",
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
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph MIS。")
        datasets = [dataset]

    all_results = []

    for dataset in datasets:
        try:
            result = solve_one_dataset(
                dataset=dataset,
                init_mode=args.init_mode,
                max_steps=args.max_steps,
                tabu_tenure=args.tabu_tenure,
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
                "mis_size": None,
                "violation_edges": None,
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
            "mis_size",
            "violation_edges",
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