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

    return adj


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


def solve_mis_with_scip(adj, time_limit=3600.0, verbose=False):
    """
    使用 SCIP 精确求解 MIS:
        max sum_i x_i
        s.t. x_u + x_v <= 1, for each edge (u,v)
        x_i in {0,1}
    默认每个图时间限制 3600 秒
    """
    try:
        from pyscipopt import Model, quicksum
    except ImportError as e:
        raise ImportError("未安装 pyscipopt，无法使用 SCIP。") from e

    n = len(adj)
    model = Model("MIS")

    if not verbose:
        model.hideOutput()

    # 每个图默认限制 3600 秒
    model.setRealParam("limits/time", float(time_limit))

    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(n)]

    for u in range(n):
        for v in adj[u]:
            if u < v:
                model.addCons(x[u] + x[v] <= 1)

    model.setObjective(quicksum(x[i] for i in range(n)), "maximize")

    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    sol = model.getBestSol()
    if sol is None:
        selected = []
    else:
        selected = [i for i in range(n) if model.getSolVal(sol, x[i]) > 0.5]

    return {
        "selected_nodes": selected,
        "solve_time_sec": elapsed,
    }


def solve_mis_exact(adj, solver="auto", time_limit=3600.0, verbose=False):
    """
    auto: 优先 SCIP
    """
    last_error = None

    if solver in ["auto", "scip"]:
        try:
            return solve_mis_with_scip(adj, time_limit=time_limit, verbose=verbose)
        except Exception as e:
            last_error = e
            if solver == "scip":
                raise

    raise RuntimeError(f"没有可用的 exact solver。最后一次报错：{last_error}")


def solve_one_dataset(dataset, solver="auto", time_limit=3600.0, verbose=False):
    """
    求解单个数据集，并输出与 greedy 相同格式的结果
    """
    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    adj = build_adj_from_graph(graph)

    solve_result = solve_mis_exact(
        adj=adj,
        solver=solver,
        time_limit=time_limit,
        verbose=verbose,
    )

    mis_nodes = solve_result["selected_nodes"]
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
        "time_sec": solve_result["solve_time_sec"],
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(graph.e[0])}")
    print(f"MIS size        : {mis_size}")
    print(f"Violation edges : {violation_edges}")
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
    parser = argparse.ArgumentParser(description="Exact Solver baseline for MIS")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="单个数据集名，例如 Graph_Cora；不填则遍历所有 graph 数据集",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        choices=["auto", "scip", "pulp"],
        help="求解器选择：auto 优先 SCIP",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=3600.0,
        help="每个数据集的时间限制（秒），默认 3600 秒",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否显示求解器详细日志",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="mis_scip_results.csv",
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
                solver=args.solver,
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