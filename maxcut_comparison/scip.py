import argparse
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

    edges = sorted(list(set(edges)))
    return edges


def evaluate_cut(graph, part):
    """
    根据原图统计 cut 值，并检查划分是否合法
    """
    if any(x not in (0, 1) for x in part):
        return 0, False

    cut_size = 0
    for edge in graph.e[0]:
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            continue
        if part[u] != part[v]:
            cut_size += 1

    return cut_size, True


def solve_maxcut_with_scip(num_nodes, edges, time_limit=3600.0, verbose=False):
    """
    使用 SCIP 精确求解 MaxCut

    变量：
        x_u in {0,1}     表示点 u 属于哪一侧
        y_uv in {0,1}    表示边 (u,v) 是否被切开

    约束：
        y_uv <= x_u + x_v
        y_uv <= 2 - x_u - x_v
        y_uv >= x_u - x_v
        y_uv >= x_v - x_u

    目标：
        max sum y_uv
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError as e:
        raise ImportError("未安装 pyscipopt，无法使用 SCIP。") from e

    model = Model("MaxCut")

    if not verbose:
        model.hideOutput()

    model.setRealParam("limits/time", float(time_limit))

    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(num_nodes)]
    y = {}

    for (u, v) in edges:
        y[(u, v)] = model.addVar(vtype="B", name=f"y_{u}_{v}")

    for (u, v) in edges:
        # y_uv = |x_u - x_v| 的线性化
        model.addCons(y[(u, v)] <= x[u] + x[v])
        model.addCons(y[(u, v)] <= 2 - x[u] - x[v])
        model.addCons(y[(u, v)] >= x[u] - x[v])
        model.addCons(y[(u, v)] >= x[v] - x[u])

    model.setObjective(quicksum(y[(u, v)] for (u, v) in edges), "maximize")

    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    sol = model.getBestSol()
    if sol is None:
        part = [0] * num_nodes
        cut_size = 0
    else:
        part = [1 if model.getSolVal(sol, x[i]) > 0.5 else 0 for i in range(num_nodes)]
        cut_size = sum(1 for (u, v) in edges if part[u] != part[v])

    return {
        "part": part,
        "cut_size": cut_size,
        "solve_time_sec": elapsed,
    }


def solve_one_dataset(dataset, time_limit=3600.0, verbose=False):
    """
    求解单个数据集
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    edges = build_edge_list_from_graph(graph)

    solve_result = solve_maxcut_with_scip(
        num_nodes=graph.num_v,
        edges=edges,
        time_limit=time_limit,
        verbose=verbose,
    )

    part = solve_result["part"]
    cut_size, is_valid = evaluate_cut(graph, part)

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(edges),
        "cut_size": cut_size,
        "is_valid": is_valid,
        "time_sec": solve_result["solve_time_sec"],
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(edges)}")
    print(f"Cut size        : {cut_size}")
    print(f"Valid           : {is_valid}")
    print(f"Time (sec)      : {solve_result['solve_time_sec']:.6f}")
    print("=" * 80)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():
    parser = argparse.ArgumentParser(description="SCIP Exact Solver for MaxCut")
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
        default="maxcut_scip_results.csv",
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
            raise ValueError(f"{args.dataset} 不是 graph 数据集，不能用于 graph MaxCut。")
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
                "cut_size": None,
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
            "cut_size",
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