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
    从图中提取无向边列表，并去除自环和重复边
    """

    edge_set = set()

    for edge in graph.e[0]:

        if len(edge) != 2:
            raise ValueError(f"发现非二元边 {edge}，这不是普通 graph 数据。")

        u, v = int(edge[0]), int(edge[1])

        if u == v:
            continue

        if u > v:
            u, v = v, u

        edge_set.add((u, v))

    edge_list = sorted(list(edge_set))

    return edge_list


def evaluate_maxkcut(edge_list, part, k):
    """
    统计 Max-k-Cut 的 cut 值，并检查划分是否合法
    """

    if any((x < 0 or x >= k) for x in part):
        return 0, False

    cut_size = 0

    for u, v in edge_list:

        if part[u] != part[v]:
            cut_size += 1

    return cut_size, True


def solve_maxkcut_by_scip(graph, k=4, time_limit=3600, gap_limit=None):
    """
    使用 SCIP 求解 Max-k-Cut。

    建模方式：
        x[i, c] = 1 表示节点 i 被分到第 c 个分区；
        y[e, c] = 1 表示边 e 的两个端点都被分到第 c 个分区。

    目标：
        最小化同分区边数量，
        等价于最大化跨分区边数量。
    """

    try:
        from pyscipopt import Model, quicksum
    except ImportError:
        raise ImportError(
            "未安装 pyscipopt，无法运行 SCIP。请先安装 pyscipopt。"
        )

    if k <= 1:
        raise ValueError("k 必须大于 1。")

    n = graph.num_v

    edge_list = build_edge_list_from_graph(graph)

    model = Model("MaxKCut_SCIP")

    model.hideOutput()

    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))

    if gap_limit is not None:
        model.setRealParam("limits/gap", float(gap_limit))

    x = {}

    for i in range(n):

        for c in range(k):

            x[i, c] = model.addVar(
                vtype="B",
                name=f"x_{i}_{c}",
            )

    y = {}

    for e_idx, (u, v) in enumerate(edge_list):

        for c in range(k):

            y[e_idx, c] = model.addVar(
                vtype="B",
                name=f"y_{e_idx}_{c}",
            )

    for i in range(n):

        model.addCons(
            quicksum(x[i, c] for c in range(k)) == 1,
            name=f"one_hot_{i}",
        )

    for e_idx, (u, v) in enumerate(edge_list):

        for c in range(k):

            model.addCons(
                y[e_idx, c] <= x[u, c],
                name=f"y_le_xu_{e_idx}_{c}",
            )

            model.addCons(
                y[e_idx, c] <= x[v, c],
                name=f"y_le_xv_{e_idx}_{c}",
            )

            model.addCons(
                y[e_idx, c] >= x[u, c] + x[v, c] - 1,
                name=f"y_ge_and_{e_idx}_{c}",
            )

    same_color_edges = quicksum(
        y[e_idx, c]
        for e_idx in range(len(edge_list))
        for c in range(k)
    )

    model.setObjective(
        same_color_edges,
        "minimize",
    )

    start = time.time()

    model.optimize()

    elapsed = time.time() - start

    status = model.getStatus()

    sol = model.getBestSol()

    if sol is None:

        return {
            "cut_size": None,
            "same_color_edges": None,
            "is_valid": False,
            "status": str(status),
            "time_sec": elapsed,
            "gap": None,
        }

    part = []

    for i in range(n):

        values = [
            model.getSolVal(sol, x[i, c])
            for c in range(k)
        ]

        chosen_color = max(
            range(k),
            key=lambda c: values[c],
        )

        part.append(chosen_color)

    cut_size, is_valid = evaluate_maxkcut(
        edge_list,
        part,
        k,
    )

    same_color_num = len(edge_list) - cut_size

    try:
        gap = model.getGap()
    except Exception:
        gap = None

    return {
        "cut_size": cut_size,
        "same_color_edges": same_color_num,
        "is_valid": is_valid,
        "status": str(status),
        "time_sec": elapsed,
        "gap": gap,
    }


def solve_one_dataset(dataset, k=4, time_limit=3600, gap_limit=None):
    """
    对单个数据集使用 SCIP 求解一次 Max-k-Cut
    """

    graph = from_file_to_graph(
        dataset.path,
        reset_vertex_index=True,
        remove_self_loops=True,
    )

    edge_list = build_edge_list_from_graph(graph)

    scip_result = solve_maxkcut_by_scip(
        graph,
        k=k,
        time_limit=time_limit,
        gap_limit=gap_limit,
    )

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "dataset_type": dataset.type,
        "file_path": dataset.path,
        "num_nodes": graph.num_v,
        "num_edges": len(edge_list),
        "k": k,
        "cut_size": scip_result["cut_size"],
        "same_color_edges": scip_result["same_color_edges"],
        "is_valid": scip_result["is_valid"],
        "status": scip_result["status"],
        "gap": scip_result["gap"],
        "time_limit_sec": time_limit,
        "time_sec": scip_result["time_sec"],
    }

    print("=" * 80)
    print(f"Dataset         : {dataset.name}")
    print(f"Path            : {dataset.path}")
    print(f"Nodes           : {graph.num_v}")
    print(f"Edges           : {len(edge_list)}")
    print(f"k               : {k}")
    print(f"Cut Size        : {scip_result['cut_size']}")
    print(f"Same Color Edges: {scip_result['same_color_edges']}")
    print(f"Valid           : {scip_result['is_valid']}")
    print(f"SCIP Status     : {scip_result['status']}")
    print(f"SCIP Gap        : {scip_result['gap']}")
    print(f"Time Limit      : {time_limit}")
    print(f"Time (sec)      : {scip_result['time_sec']:.6f}")
    print("=" * 80)

    return result


def get_graph_datasets():
    """
    只取 graph 类型数据集
    """
    return [d for d in Datasets if d.type == "graph"]


def parse_args():

    parser = argparse.ArgumentParser(
        description="SCIP baseline for Max-k-Cut"
    )

    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="Max-k-Cut 中的分区数量，例如 4 5 6",
    )

    parser.add_argument(
        "--time_limit",
        type=float,
        default=3600,
        help="SCIP 单个数据集求解时间上限，单位为秒，默认 3600 秒",
    )

    parser.add_argument(
        "--gap_limit",
        type=float,
        default=None,
        help="SCIP 相对 gap 停止条件，例如 0.01；不设置则为 None",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="maxkcut_scip_k456_results.csv",
        help="保存的 csv 文件名",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    for k in args.k_values:
        if k <= 1:
            raise ValueError("k 必须大于 1。")

    datasets = get_graph_datasets()

    all_results = []

    for k in args.k_values:

        print("\n" + "#" * 80)
        print(f"Start SCIP Max-k-Cut experiments with k = {k}")
        print("#" * 80)

        for dataset in datasets:

            try:

                result = solve_one_dataset(
                    dataset,
                    k=k,
                    time_limit=args.time_limit,
                    gap_limit=args.gap_limit,
                )

                all_results.append(result)

            except Exception as e:

                print("=" * 80)
                print(f"[ERROR] Dataset {dataset.name}, k={k} failed: {e}")
                print("=" * 80)

                all_results.append({
                    "dataset_enum": dataset.name,
                    "dataset_value": dataset.value,
                    "dataset_type": dataset.type,
                    "file_path": dataset.path,
                    "num_nodes": None,
                    "num_edges": None,
                    "k": k,
                    "cut_size": None,
                    "same_color_edges": None,
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
            "k",
            "cut_size",
            "same_color_edges",
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