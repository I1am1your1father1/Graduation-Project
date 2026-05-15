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


def evaluate_coloring(graph, colors):
    """
    评估着色结果：
    - num_colors: 实际使用颜色数
    - conflict_edges: 冲突边数
    - is_valid: 是否合法
    """
    if colors is None:
        return None, None, False

    if any(c < 0 for c in colors):
        return None, None, False

    num_colors = len(set(colors)) if len(colors) > 0 else 0

    conflict_edges = 0

    for edge in graph.e[0]:
        u, v = int(edge[0]), int(edge[1])

        if u == v:
            continue

        if colors[u] == colors[v]:
            conflict_edges += 1

    is_valid = (conflict_edges == 0)

    return num_colors, conflict_edges, is_valid


def solve_coloring_with_scip(num_nodes, edges, time_limit=3600.0, gap_limit=None, verbose=False):
    """
    使用纯 SCIP 求解 Graph Coloring。

    不使用 DSATUR；
    不使用 fallback_colors；
    颜色上界直接设为 num_nodes。

    变量：
        x[v,c] in {0,1}   点 v 是否使用颜色 c
        y[c]   in {0,1}   颜色 c 是否被启用

    目标：
        min sum_c y[c]

    约束：
        1) 每个点恰好一种颜色
        2) 相邻点不能同色
        3) 若点使用颜色 c，则颜色 c 必须被启用
        4) 对称性削弱：y[c] >= y[c+1]
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError as e:
        raise ImportError("未安装 pyscipopt，无法使用 SCIP。") from e

    if num_nodes < 0:
        raise ValueError("num_nodes 不能为负数。")

    if num_nodes == 0:
        return {
            "colors": [],
            "solve_time_sec": 0.0,
            "status": "empty_graph",
            "gap": 0.0,
        }

    model = Model("Pure_SCIP_GraphColoring")

    if not verbose:
        model.hideOutput()

    model.setRealParam("limits/time", float(time_limit))

    if gap_limit is not None:
        model.setRealParam("limits/gap", float(gap_limit))

    # 纯 SCIP：颜色上界直接设为节点数
    k = 100

    x = {}
    y = {}

    for c in range(k):
        y[c] = model.addVar(
            vtype="B",
            name=f"y_{c}",
        )

    for v in range(num_nodes):
        for c in range(k):
            x[(v, c)] = model.addVar(
                vtype="B",
                name=f"x_{v}_{c}",
            )

    # 每个节点恰好选择一种颜色
    for v in range(num_nodes):
        model.addCons(
            quicksum(x[(v, c)] for c in range(k)) == 1,
            name=f"one_color_{v}",
        )

    # 相邻节点不能使用同一种颜色
    for u, v in edges:
        for c in range(k):
            model.addCons(
                x[(u, c)] + x[(v, c)] <= 1,
                name=f"edge_{u}_{v}_color_{c}",
            )

    # 如果节点 v 使用颜色 c，则颜色 c 必须被启用
    for v in range(num_nodes):
        for c in range(k):
            model.addCons(
                x[(v, c)] <= y[c],
                name=f"use_color_{v}_{c}",
            )

    # 对称性削弱：优先使用编号较小的颜色
    for c in range(k - 1):
        model.addCons(
            y[c] >= y[c + 1],
            name=f"symmetry_y_{c}",
        )

    # 对称性削弱：固定第 0 个节点使用颜色 0
    model.addCons(
        x[(0, 0)] == 1,
        name="fix_node_0_color_0",
    )

    model.setObjective(
        quicksum(y[c] for c in range(k)),
        "minimize",
    )

    start = time.time()

    model.optimize()

    elapsed = time.time() - start

    status = model.getStatus()

    try:
        gap = model.getGap()
    except Exception:
        gap = None

    sol = model.getBestSol()

    if sol is None:
        return {
            "colors": None,
            "solve_time_sec": elapsed,
            "status": str(status),
            "gap": gap,
        }

    colors = [-1] * num_nodes

    for v in range(num_nodes):
        for c in range(k):
            if model.getSolVal(sol, x[(v, c)]) > 0.5:
                colors[v] = c
                break

    return {
        "colors": colors,
        "solve_time_sec": elapsed,
        "status": str(status),
        "gap": gap,
    }


def solve_one_dataset(dataset, time_limit=3600.0, gap_limit=None, verbose=False):
    """
    求解单个数据集
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    edges = build_edge_list_from_graph(graph)

    solve_result = solve_coloring_with_scip(
        num_nodes=graph.num_v,
        edges=edges,
        time_limit=time_limit,
        gap_limit=gap_limit,
        verbose=verbose,
    )

    colors = solve_result["colors"]

    num_colors, conflict_edges, is_valid = evaluate_coloring(
        graph,
        colors,
    )

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(edges),
        "num_colors": num_colors,
        "conflict_edges": conflict_edges,
        "is_valid": is_valid,
        "status": solve_result["status"],
        "gap": solve_result["gap"],
        "time_limit_sec": time_limit,
        "time_sec": solve_result["solve_time_sec"],
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(edges)}")
    print(f"Num colors      : {num_colors}")
    print(f"Conflict edges  : {conflict_edges}")
    print(f"Valid           : {is_valid}")
    print(f"SCIP Status     : {solve_result['status']}")
    print(f"SCIP Gap        : {solve_result['gap']}")
    print(f"Time Limit      : {time_limit}")
    print(f"Time (sec)      : {solve_result['solve_time_sec']:.6f}")
    print("=" * 80)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集，并去掉两个 Amazon 数据集和 Pubmed 数据集
    """
    exclude_datasets = {
        Datasets.Graph_Amazon_PC,
        Datasets.Graph_Amazon_Photo,
        Datasets.Graph_Pubmed,
    }

    return [
        d for d in Datasets
        if d.type == "graph" and d not in exclude_datasets
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure SCIP Solver for Graph Coloring"
    )

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
        "--gap_limit",
        type=float,
        default=None,
        help="SCIP 相对 gap 停止条件，例如 0.01；不设置则为 None",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否显示 SCIP 日志",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="coloring_pure_scip_results.csv",
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
                time_limit=args.time_limit,
                gap_limit=args.gap_limit,
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
                "num_colors": None,
                "conflict_edges": None,
                "is_valid": False,
                "status": f"ERROR: {e}",
                "gap": None,
                "time_limit_sec": args.time_limit,
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
            "status",
            "gap",
            "time_limit_sec",
            "time_sec",
        ]
    ]

    save_dir = ROOT_DIR / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / args.save_name

    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n结果已保存到：")
    print(save_path)


if __name__ == "__main__":
    main()