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

    return adj


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


def solve_mds_with_scip(adj, time_limit=3600.0, verbose=False):
    """
    纯 SCIP 求解 MDS：
        min sum x_u
        s.t. x_u + sum_{v in N(u)} x_v >= 1,  for all u
        x_u in {0,1}
    不使用 greedy warm start，也不使用 fallback 初始解
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError as e:
        raise ImportError("未安装 pyscipopt，无法使用 SCIP。") from e

    n = len(adj)
    model = Model("MDS")

    if not verbose:
        model.hideOutput()

    model.setRealParam("limits/time", float(time_limit))

    x = [model.addVar(vtype="B", name=f"x_{u}") for u in range(n)]

    for u in range(n):
        model.addCons(x[u] + quicksum(x[v] for v in adj[u]) >= 1)

    model.setObjective(quicksum(x[u] for u in range(n)), "minimize")

    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    sol = model.getBestSol()

    if sol is None:
        raise RuntimeError("SCIP 未返回可行解（可能超时且未找到 incumbent）。")

    selected = [model.getSolVal(sol, x[u]) > 0.5 for u in range(n)]

    return {
        "selected": selected,
        "solve_time_sec": elapsed,
    }


def solve_one_dataset(dataset, time_limit=3600.0, verbose=False):
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj = build_adj_from_graph(graph)

    solve_result = solve_mds_with_scip(
        adj=adj,
        time_limit=time_limit,
        verbose=verbose,
    )

    selected = solve_result["selected"]
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
        "time_sec": solve_result["solve_time_sec"],
    }

    print("=" * 80)
    print(f"Dataset           : {dataset.name}")
    print(f"Path              : {dataset.path}")
    print(f"Nodes             : {graph.num_v}")
    print(f"Edges             : {len(graph.e[0])}")
    print(f"MDS size          : {mds_size}")
    print(f"Undominated nodes : {undominated_nodes}")
    print(f"Valid             : {is_valid}")
    print(f"Time (sec)        : {solve_result['solve_time_sec']:.6f}")
    print("=" * 80)

    return result


def get_graph_datasets():
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="Pure SCIP exact solver for MDS")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
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
        default="mds_scip_results.csv",
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
                time_limit=args.time_limit,
                verbose=args.verbose,
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