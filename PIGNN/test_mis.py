import dgl
import torch
import random
import os
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from enum import Enum
from pathlib import Path
from collections import defaultdict
from dgl.nn.pytorch import GraphConv
from itertools import combinations
from time import time

# MacOS can have issues with MKL.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# =========================
# 0. 设备设置
# =========================

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32
print(f"Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}")

from utils import (
    generate_graph,
    get_gnn,
    run_gnn_training,
    qubo_dict_to_torch,
    gen_combinations,
    loss_func,
)


# =========================
# 1. 数据集枚举类
# =========================

class Datasets(Enum):
    Graph_Cora = "cora_graph"
    Graph_Citeseer = "citeseer_graph"
    Graph_Amazon_PC = "amazon_electronics_computers_graph"
    Graph_Amazon_Photo = "amazon_electronics_photo_graph"
    Graph_Pubmed = "pubmed_graph"
    Graph_dblp = "dblp_graph"
    Graph_bat = "bat_graph"
    Graph_eat = "eat_graph"
    Graph_uat = "uat_graph"

    @property
    def path(self):
        current_path = Path(__file__).parent
        type_path = self.value.split("_")[-1]
        data_path = os.path.join(
            current_path,
            "..",
            "data",
            type_path,
            f"{self.value}.txt",
        )
        return data_path

    @property
    def type(self):
        return self.value.split("_")[-1]


# =========================
# 2. 实验全局参数
# =========================

# 每个数据集重复运行次数
NUM_RUNS = 10

# 基础随机种子
BASE_SEED = 1

# NN learning hypers
number_epochs = int(1e5)
learning_rate = 1e-4
PROB_THRESHOLD = 0.5

# Early stopping
tol = 1e-4
patience = 100

# 需要一次性运行的全部数据集
ALL_DATASETS = [
    Datasets.Graph_bat,
    Datasets.Graph_eat,
    Datasets.Graph_uat,
    Datasets.Graph_dblp,
    Datasets.Graph_Cora,
    Datasets.Graph_Citeseer,
    Datasets.Graph_Amazon_Photo,
    Datasets.Graph_Amazon_PC,
    Datasets.Graph_Pubmed,
]


# =========================
# 3. 随机种子设置
# =========================

def set_seed(seed: int):
    """
    设置随机种子。
    每次运行使用不同 seed，使 10 次结果不同；
    但固定 BASE_SEED 后整体实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 尽量提高可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 4. 读取真实图数据集
# =========================

def load_graph_from_dataset(dataset: Datasets):
    """
    从 dataset.path 中读取图数据。

    txt 文件格式：
        第一行：节点数 边数
              如果未知，则为 ? ?
        后续每一行：一条边，例如
              35 1033
              35 103482

    注意：
        1. 第一行不是边，必须跳过；
        2. 原始节点编号可能很大，所以必须重新编号为 0,1,...,n-1；
        3. 如果第一行给出了节点数，并且边文件中没有出现所有节点，则会补充孤立点。
    """
    data_path = dataset.path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    print(f"Loading dataset: {dataset.value}")
    print(f"Dataset path: {data_path}")

    nx_temp = nx.Graph()

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) == 0:
        raise ValueError("Dataset file is empty.")

    header = lines[0].strip().replace(",", " ").split()

    declared_num_nodes = None
    declared_num_edges = None

    if len(header) >= 2:
        if header[0] != "?":
            declared_num_edges = int(header[0])
        if header[1] != "?":
            declared_num_nodes = int(header[1])

    print(f"Declared nodes: {declared_num_nodes}")
    print(f"Declared edges: {declared_num_edges}")

    for line in lines[1:]:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        line = line.replace(",", " ")
        parts = line.split()

        if len(parts) < 2:
            continue

        u, v = parts[0], parts[1]

        # 去掉自环
        if u == v:
            continue

        nx_temp.add_edge(u, v)

    if nx_temp.number_of_nodes() == 0:
        raise ValueError("Loaded graph is empty. Please check dataset format.")

    if declared_num_nodes is not None:
        current_num_nodes = nx_temp.number_of_nodes()

        if current_num_nodes < declared_num_nodes:
            num_isolated = declared_num_nodes - current_num_nodes
            print(f"Adding {num_isolated} isolated nodes according to header.")

            for i in range(num_isolated):
                nx_temp.add_node(f"__isolated_{i}")

        elif current_num_nodes > declared_num_nodes:
            print(
                f"Warning: actual nodes from edge list ({current_num_nodes}) "
                f"is larger than declared nodes ({declared_num_nodes})."
            )

    if declared_num_edges is not None:
        actual_num_edges = nx_temp.number_of_edges()

        if actual_num_edges != declared_num_edges:
            print(
                f"Warning: actual edges ({actual_num_edges}) "
                f"!= declared edges ({declared_num_edges}). "
                f"This may be caused by duplicate edges or self-loops."
            )

    # 重新编号为 0, 1, ..., n-1
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)

    # 保持节点顺序稳定
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges())

    print(
        f"Loaded graph with n={nx_graph.number_of_nodes()}, "
        f"m={nx_graph.number_of_edges()}"
    )

    return nx_graph


# =========================
# 5. 构造 MIS 的 QUBO 矩阵
# =========================

def gen_q_dict_mis(nx_G, penalty=2):
    """
    构造 MIS 的 QUBO 矩阵。

    MIS 最大独立集问题可写为最小化：
        - sum_i x_i + penalty * sum_(i,j in E) x_i x_j

    x_i = 1 表示节点 i 被选入独立集。
    """
    Q_dic = defaultdict(int)

    # 边惩罚项
    for u, v in nx_G.edges():
        Q_dic[(u, v)] = penalty

    # 对角线奖励项，鼓励选择节点
    for u in nx_G.nodes():
        Q_dic[(u, u)] = -1

    return Q_dic


# =========================
# 6. 处理 PI-GNN 输出结果
# =========================

def postprocess_gnn_mis(best_bitstring, nx_graph):
    """
    处理 PI-GNN 求解 MIS 的结果。

    best_bitstring:
        0/1 向量，1 表示节点被选入独立集。

    返回：
        size_mis: 选中节点数量
        ind_set: 选中节点集合
        number_violations: 独立集冲突边数量
    """
    if isinstance(best_bitstring, torch.Tensor):
        bitstring_list = best_bitstring.detach().cpu().long().view(-1).tolist()
    else:
        bitstring_list = list(best_bitstring)

    ind_set = set(
        node for node, value in enumerate(bitstring_list)
        if int(value) == 1
    )

    size_mis = len(ind_set)

    number_violations = 0
    for u, v in nx_graph.edges():
        if u in ind_set and v in ind_set:
            number_violations += 1

    return size_mis, ind_set, number_violations


# =========================
# 7. 单次运行 PI-GNN
# =========================

def run_one_pignn_trial(dataset: Datasets, run_id: int, seed: int):
    """
    对一个数据集运行一次 PI-GNN。
    """
    set_seed(seed)

    print("\n" + "=" * 100)
    print(f"Dataset: {dataset.name} | Run: {run_id + 1}/{NUM_RUNS} | Seed: {seed}")
    print("=" * 100)

    # 1. 加载图
    nx_graph = load_graph_from_dataset(dataset)
    n = nx_graph.number_of_nodes()
    m = nx_graph.number_of_edges()

    # 2. 构造 DGL 图
    graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
    graph_dgl = dgl.add_self_loop(graph_dgl)
    graph_dgl = graph_dgl.to(TORCH_DEVICE)

    # 3. 构造 QUBO 矩阵
    q_torch = qubo_dict_to_torch(
        nx_graph,
        gen_q_dict_mis(nx_graph),
        torch_dtype=TORCH_DTYPE,
        torch_device=TORCH_DEVICE,
    )

    # 4. PI-GNN 超参数
    dim_embedding = int(np.sqrt(n))
    hidden_dim = int(dim_embedding / 2)

    if dim_embedding < 1:
        dim_embedding = 1
    if hidden_dim < 1:
        hidden_dim = 1

    opt_params = {
        "lr": learning_rate,
    }

    gnn_hypers = {
        "dim_embedding": dim_embedding,
        "hidden_dim": hidden_dim,
        "dropout": 0.0,
        "number_classes": 1,
        "prob_threshold": PROB_THRESHOLD,
        "number_epochs": number_epochs,
        "tolerance": tol,
        "patience": patience,
    }

    # 每次运行都重新初始化模型、embedding、optimizer
    net, embed, optimizer = get_gnn(
        n,
        gnn_hypers,
        opt_params,
        TORCH_DEVICE,
        TORCH_DTYPE,
    )

    # 5. 运行 PI-GNN
    print("Running PI-GNN...")
    start_time = time()

    _, epoch, final_bitstring, best_bitstring = run_gnn_training(
        q_torch,
        graph_dgl,
        net,
        embed,
        optimizer,
        gnn_hypers["number_epochs"],
        gnn_hypers["tolerance"],
        gnn_hypers["patience"],
        gnn_hypers["prob_threshold"],
    )

    train_time = time() - start_time

    # 6. 处理最优 bitstring
    size_mis, ind_set, number_violations = postprocess_gnn_mis(
        best_bitstring,
        nx_graph,
    )

    is_valid = number_violations == 0

    # 7. 计算 final loss，失败时不影响主流程
    try:
        final_loss = loss_func(
            final_bitstring.float().to(TORCH_DEVICE),
            q_torch,
        )
        final_loss_value = float(final_loss.detach().cpu().item())
    except Exception:
        final_loss_value = None

    print(
        f"PI-GNN result | "
        f"MIS size: {size_mis} | "
        f"Violation edges: {number_violations} | "
        f"Valid: {is_valid} | "
        f"Epoch: {epoch} | "
        f"Time: {train_time:.3f}s"
    )

    result = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "run": run_id + 1,
        "seed": seed,
        "num_nodes": n,
        "num_edges": m,
        "mis_size": int(size_mis),
        "violation_edges": int(number_violations),
        "is_valid": bool(is_valid),
        "epoch": int(epoch),
        "time_sec": float(train_time),
        "final_loss": final_loss_value,
    }

    # 释放显存
    del net, embed, optimizer, graph_dgl, q_torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =========================
# 8. 汇总单个数据集的 10 次结果
# =========================

def summarize_dataset_results(dataset: Datasets, dataset_results):
    """
    汇总一个数据集 10 次运行结果。

    best_mis_size：
        优先在合法解中选最大 MIS；
        如果 10 次都不合法，则从所有结果中选 mis_size 最大的结果。

    avg_mis_size：
        10 次运行的平均 MIS 大小。
    """
    df = pd.DataFrame(dataset_results)

    # 去掉失败的运行
    df = df[df["mis_size"].notna()].copy()

    if len(df) == 0:
        return {
            "dataset_enum": dataset.name,
            "dataset_value": dataset.value,
            "num_nodes": None,
            "num_edges": None,
            "best_mis_size": None,
            "best_run": None,
            "best_seed": None,
            "best_violation_edges": None,
            "best_is_valid": False,
            "avg_mis_size": None,
            "std_mis_size": None,
            "avg_valid_mis_size": None,
            "std_valid_mis_size": None,
            "valid_runs": 0,
            "avg_violation_edges": None,
            "avg_time_sec": None,
            "std_time_sec": None,
            "best_time_sec": None,
        }

    valid_df = df[df["is_valid"] == True]

    if len(valid_df) > 0:
        best_row = valid_df.loc[valid_df["mis_size"].idxmax()]
    else:
        best_row = df.loc[df["mis_size"].idxmax()]

    if len(valid_df) > 0:
        avg_valid_mis_size = float(valid_df["mis_size"].mean())
        std_valid_mis_size = float(valid_df["mis_size"].std(ddof=0))
    else:
        avg_valid_mis_size = None
        std_valid_mis_size = None

    summary = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "num_nodes": int(df["num_nodes"].iloc[0]),
        "num_edges": int(df["num_edges"].iloc[0]),

        # 10 次中最好结果
        "best_mis_size": int(best_row["mis_size"]),
        "best_run": int(best_row["run"]),
        "best_seed": int(best_row["seed"]),
        "best_violation_edges": int(best_row["violation_edges"]),
        "best_is_valid": bool(best_row["is_valid"]),

        # 所有运行的平均结果
        "avg_mis_size": float(df["mis_size"].mean()),
        "std_mis_size": float(df["mis_size"].std(ddof=0)),

        # 只统计合法运行的平均结果
        "avg_valid_mis_size": avg_valid_mis_size,
        "std_valid_mis_size": std_valid_mis_size,

        # 合法性统计
        "valid_runs": int(df["is_valid"].sum()),
        "avg_violation_edges": float(df["violation_edges"].mean()),

        # 时间统计
        "avg_time_sec": float(df["time_sec"].mean()),
        "std_time_sec": float(df["time_sec"].std(ddof=0)),
        "best_time_sec": float(best_row["time_sec"]),
    }

    return summary


# =========================
# 9. 主函数：一次运行得到全部结果
# =========================

def main():
    all_run_results = []
    all_summary_results = []

    for dataset_idx, dataset in enumerate(ALL_DATASETS):
        dataset_results = []

        print("\n" + "#" * 100)
        print(f"Start dataset: {dataset.name} ({dataset_idx + 1}/{len(ALL_DATASETS)})")
        print("#" * 100)

        for run_id in range(NUM_RUNS):
            seed = BASE_SEED + dataset_idx * 1000 + run_id

            try:
                result = run_one_pignn_trial(
                    dataset=dataset,
                    run_id=run_id,
                    seed=seed,
                )

            except Exception as e:
                print("\n" + "=" * 100)
                print(f"[ERROR] Dataset {dataset.name}, Run {run_id + 1} failed: {e}")
                print("=" * 100)

                result = {
                    "dataset_enum": dataset.name,
                    "dataset_value": dataset.value,
                    "run": run_id + 1,
                    "seed": seed,
                    "num_nodes": None,
                    "num_edges": None,
                    "mis_size": None,
                    "violation_edges": None,
                    "is_valid": False,
                    "epoch": None,
                    "time_sec": None,
                    "final_loss": None,
                    "error": str(e),
                }

            dataset_results.append(result)
            all_run_results.append(result)

        # 当前数据集 10 次运行结束后汇总
        summary = summarize_dataset_results(dataset, dataset_results)
        all_summary_results.append(summary)

        print("\n" + "-" * 100)
        print(f"Summary for {dataset.name}")
        print(
            f"Best MIS size: {summary['best_mis_size']} | "
            f"Average MIS size: {summary['avg_mis_size']} ± {summary['std_mis_size']} | "
            f"Valid runs: {summary['valid_runs']}/{NUM_RUNS} | "
            f"Average time: {summary['avg_time_sec']}s"
        )
        print("-" * 100)

    # =========================
    # 10. 保存结果
    # =========================

    current_path = Path(__file__).parent
    save_dir = current_path / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    detail_path = save_dir / "mis_pignn_run_details.csv"
    summary_path = save_dir / "mis_pignn_summary.csv"

    df_detail = pd.DataFrame(all_run_results)
    df_summary = pd.DataFrame(all_summary_results)

    # 调整列顺序
    detail_columns = [
        "dataset_enum",
        "dataset_value",
        "run",
        "seed",
        "num_nodes",
        "num_edges",
        "mis_size",
        "violation_edges",
        "is_valid",
        "epoch",
        "time_sec",
        "final_loss",
        "error",
    ]
    detail_columns = [c for c in detail_columns if c in df_detail.columns]
    df_detail = df_detail[detail_columns]

    summary_columns = [
        "dataset_enum",
        "dataset_value",
        "num_nodes",
        "num_edges",
        "best_mis_size",
        "best_run",
        "best_seed",
        "best_violation_edges",
        "best_is_valid",
        "avg_mis_size",
        "std_mis_size",
        "avg_valid_mis_size",
        "std_valid_mis_size",
        "valid_runs",
        "avg_violation_edges",
        "avg_time_sec",
        "std_time_sec",
        "best_time_sec",
    ]
    summary_columns = [c for c in summary_columns if c in df_summary.columns]
    df_summary = df_summary[summary_columns]

    df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("全部 PI-GNN 实验完成。")
    print(f"每次运行详细结果已保存到：{detail_path}")
    print(f"每个数据集汇总结果已保存到：{summary_path}")
    print("=" * 100)

    print("\n汇总结果：")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()