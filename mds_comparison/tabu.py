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


def greedy_mds_naive(adj):
    n = len(adj)
    closed = [set([u]) | adj[u] for u in range(n)]

    undominated = set(range(n))
    selected = [False] * n

    while undominated:
        best_u = None
        best_gain = None
        best_deg = None

        for u in range(n):
            gain = len(closed[u] & undominated)
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

        selected[best_u] = True
        undominated -= closed[best_u]

    return selected


def random_mds(adj, seed=42):
    rng = random.Random(seed)
    n = len(adj)
    closed = [set([u]) | adj[u] for u in range(n)]

    undominated = set(range(n))
    selected = [False] * n

    while undominated:
        target = rng.choice(tuple(undominated))
        chosen = rng.choice(list(closed[target]))
        selected[chosen] = True
        undominated -= closed[chosen]

    return selected


def compute_dom_count(selected, closed):
    n = len(selected)
    dom_count = [0] * n
    for u, flag in enumerate(selected):
        if flag:
            for w in closed[u]:
                dom_count[w] += 1
    return dom_count


def prune_redundant(selected, adj, seed=42):
    rng = random.Random(seed)
    n = len(adj)
    closed = [set([u]) | adj[u] for u in range(n)]
    dom_count = compute_dom_count(selected, closed)

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


def tabu_search_mds(adj, init_mode="greedy", max_steps=3000, tabu_tenure=7, seed=42):
    """
    用惩罚目标做简单 tabu：
        obj = |S| + penalty * undominated_count
    目标越小越好。
    """
    rng = random.Random(seed)
    n = len(adj)
    closed = [set([u]) | adj[u] for u in range(n)]
    penalty = n + 1

    if init_mode == "greedy":
        selected = greedy_mds_naive(adj)
    elif init_mode == "random":
        selected = random_mds(adj, seed=seed)
    else:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    selected = prune_redundant(selected, adj, seed=seed)

    dom_count = compute_dom_count(selected, closed)
    size = sum(1 for x in selected if x)
    undom_count = sum(1 for x in dom_count if x == 0)

    best_feasible = selected.copy() if undom_count == 0 else None
    best_feasible_size = size if undom_count == 0 else float("inf")

    tabu_until = [0] * n

    for step in range(1, max_steps + 1):
        best_move = None  # (new_obj, new_undom, new_size, u)
        current_obj = size + penalty * undom_count

        for u in range(n):
            if selected[u]:
                newly_undominated = sum(1 for w in closed[u] if dom_count[w] == 1)
                new_undom = undom_count + newly_undominated
                new_size = size - 1
            else:
                newly_dominated = sum(1 for w in closed[u] if dom_count[w] == 0)
                new_undom = undom_count - newly_dominated
                new_size = size + 1

            new_obj = new_size + penalty * new_undom

            is_tabu = step < tabu_until[u]
            aspiration = (new_undom == 0 and new_size < best_feasible_size)

            if is_tabu and not aspiration:
                continue

            candidate = (new_obj, new_undom, new_size, u)

            if best_move is None:
                best_move = candidate
            else:
                if candidate[:4] < best_move[:4]:
                    best_move = candidate

        if best_move is None:
            break

        _, new_undom, new_size, u = best_move

        if selected[u]:
            selected[u] = False
            for w in closed[u]:
                dom_count[w] -= 1
        else:
            selected[u] = True
            for w in closed[u]:
                dom_count[w] += 1

        size = new_size
        undom_count = new_undom

        tabu_until[u] = step + tabu_tenure + rng.randint(0, 2)

        if undom_count == 0 and size < best_feasible_size:
            candidate_sel = selected.copy()
            candidate_sel = prune_redundant(candidate_sel, adj, seed=seed + step)
            cand_size, cand_undom, cand_valid = evaluate_mds(adj, candidate_sel)
            if cand_valid and cand_size < best_feasible_size:
                best_feasible = candidate_sel
                best_feasible_size = cand_size

    if best_feasible is None:
        best_feasible = greedy_mds_naive(adj)

    best_feasible = prune_redundant(best_feasible, adj, seed=seed)
    return best_feasible


def solve_one_dataset(dataset, init_mode="greedy", max_steps=3000, tabu_tenure=7, seed=42):
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj, _ = build_adj_from_graph(graph)

    start = time.time()
    selected = tabu_search_mds(
        adj=adj,
        init_mode=init_mode,
        max_steps=max_steps,
        tabu_tenure=tabu_tenure,
        seed=seed,
    )
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
    parser = argparse.ArgumentParser(description="Tabu baseline for MDS")
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
        default=3000,
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
        default="mds_tabu_results.csv",
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