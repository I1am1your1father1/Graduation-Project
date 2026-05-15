import dgl
import torch
import random
import os
import numpy as np
import networkx as nx
import pandas as pd

from enum import Enum
from pathlib import Path
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

# MDS loss 权重
OBJ_COEF = 1.0
CONS_COEF = 10.0
GINI_COEF = 5.0
GINI_WARMUP = 500

# 是否对 PI-GNN 输出结果进行合法性修复
# True：如果阈值化结果不是合法支配集，则用贪心方式补点
REPAIR = True

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
# 5. 构造 MDS 的闭邻域矩阵
# =========================

def build_closed_neighborhood_matrix(nx_graph):
    """
    构造 MDS 的闭邻域稀疏矩阵 B。

    B[v, u] = 1 表示节点 u 属于节点 v 的闭邻域 N[v]。
    coverage = B @ x
    coverage[v] 表示节点 v 是否被支配。
    """
    n = nx_graph.number_of_nodes()

    row = []
    col = []

    # 每个节点自己支配自己
    for u in nx_graph.nodes():
        row.append(u)
        col.append(u)

    # 邻接节点之间相互支配
    for u, v in nx_graph.edges():
        row.append(u)
        col.append(v)

        row.append(v)
        col.append(u)

    indices = torch.tensor(
        [row, col],
        dtype=torch.long,
        device=TORCH_DEVICE,
    )

    values = torch.ones(
        len(row),
        dtype=TORCH_DTYPE,
        device=TORCH_DEVICE,
    )

    B = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n, n),
        dtype=TORCH_DTYPE,
        device=TORCH_DEVICE,
    )

    return B.coalesce()


# =========================
# 6. MDS 的连续松弛损失函数
# =========================

def loss_func_mds(probs, B, epoch):
    """
    PI-GNN 求解 MDS 的损失函数。

    MDS:
        min sum_i x_i
        s.t. x_i + sum_{j in N(i)} x_j >= 1, for all i

    probs[i] 表示节点 i 被选入支配集的概率。
    """
    coverage = torch.sparse.mm(B, probs.unsqueeze(1)).squeeze()

    # 目标项：支配集规模越小越好
    loss_obj = probs.sum()

    # 约束项：只惩罚未被覆盖的节点
    violation = torch.relu(1.0 - coverage)
    loss_cons = violation.pow(2).sum()

    # 离散化项：鼓励 probs 接近 0 或 1
    loss_gini = (1.0 - (2.0 * probs - 1.0).pow(2)).sum()

    gini_coef = min(GINI_COEF, GINI_COEF * epoch / GINI_WARMUP)

    loss = (
        OBJ_COEF * loss_obj
        + CONS_COEF * loss_cons
        + gini_coef * loss_gini
    )

    return loss, loss_obj, loss_cons, loss_gini, coverage, gini_coef


# =========================
# 7. 处理 PI-GNN 输出结果
# =========================

def postprocess_gnn_mds(best_bitstring, nx_graph):
    """
    处理 PI-GNN 求解 MDS 的结果。

    best_bitstring:
        0/1 向量，1 表示节点被选入支配集。

    返回：
        size_mds: 选中节点数量
        dom_set: 选中节点集合
        undominated_nodes: 未被支配节点集合
        is_valid: 是否为合法支配集
    """
    if isinstance(best_bitstring, torch.Tensor):
        bitstring_list = best_bitstring.detach().cpu().long().view(-1).tolist()
    else:
        bitstring_list = list(best_bitstring)

    dom_set = set(
        node for node, value in enumerate(bitstring_list)
        if int(value) == 1
    )

    dominated = set(dom_set)

    for u in dom_set:
        dominated.update(nx_graph.neighbors(u))

    all_nodes = set(nx_graph.nodes())
    undominated_nodes = sorted(list(all_nodes - dominated))

    size_mds = len(dom_set)
    is_valid = len(undominated_nodes) == 0
    coverage_ratio = len(dominated) / nx_graph.number_of_nodes() if nx_graph.number_of_nodes() > 0 else 0.0

    return size_mds, dom_set, undominated_nodes, is_valid, coverage_ratio


def repair_gnn_mds(best_bitstring, nx_graph):
    """
    对 PI-GNN 输出的 MDS 解进行修复。

    如果存在未被支配节点，则每一步选择能新支配最多未支配节点的点加入支配集。
    """
    if isinstance(best_bitstring, torch.Tensor):
        repaired = best_bitstring.detach().cpu().long().view(-1).clone()
    else:
        repaired = torch.tensor(best_bitstring, dtype=torch.long)

    n = nx_graph.number_of_nodes()

    closed_neighborhood = []
    for u in range(n):
        nbh = set(nx_graph.neighbors(u))
        nbh.add(u)
        closed_neighborhood.append(nbh)

    added_nodes = []

    while True:
        _, _, undominated_nodes, is_valid, _ = postprocess_gnn_mds(
            repaired,
            nx_graph,
        )

        if is_valid:
            break

        undominated = set(undominated_nodes)

        best_u = None
        best_gain = None
        best_degree = None

        candidate_nodes = set()

        for v in undominated:
            candidate_nodes.update(closed_neighborhood[v])

        for u in candidate_nodes:
            gain = len(closed_neighborhood[u] & undominated)
            degree_u = nx_graph.degree[u]

            if best_u is None:
                best_u = u
                best_gain = gain
                best_degree = degree_u
            else:
                if gain > best_gain:
                    best_u = u
                    best_gain = gain
                    best_degree = degree_u
                elif gain == best_gain:
                    if degree_u > best_degree:
                        best_u = u
                        best_gain = gain
                        best_degree = degree_u
                    elif degree_u == best_degree and u < best_u:
                        best_u = u
                        best_gain = gain
                        best_degree = degree_u

        if best_u is None:
            best_u = min(undominated)

        if repaired[best_u].item() == 0:
            repaired[best_u] = 1
            added_nodes.append(best_u)
        else:
            break

    return repaired, added_nodes


# =========================
# 8. MDS 的 PI-GNN 训练函数
# =========================

def run_gnn_training_mds(
    B,
    dgl_graph,
    net,
    embed,
    optimizer,
    number_epochs,
    tol,
    patience,
    prob_threshold,
):
    """
    Wrapper function to run and monitor PI-GNN training for MDS.
    Includes early stopping.
    """
    # Assign variable for user reference
    inputs = embed.weight

    prev_loss = 1.
    count = 0

    n = dgl_graph.number_of_nodes()

    # initialize optimal solution
    best_bitstring = torch.zeros((n,)).type(B.dtype).to(B.device)
    best_score = n * (n + 1)

    t_gnn_start = time()

    # Training logic
    for epoch in range(number_epochs):

        # get logits/activations
        probs = net(dgl_graph, inputs)[:, 0]

        # build MDS loss
        loss, loss_obj, loss_cons, loss_gini, coverage, gini_coef = loss_func_mds(
            probs,
            B,
            epoch + 1,
        )
        loss_ = loss.detach().item()

        # Apply projection
        bitstring = (probs.detach() >= prob_threshold) * 1
        bitstring = bitstring.type(B.dtype).to(B.device)

        # 以离散解质量选择 best_bitstring
        discrete_coverage = torch.sparse.mm(B, bitstring.unsqueeze(1)).squeeze()
        undominated_num = int((discrete_coverage < 0.5).sum().detach().cpu().item())
        selected_num = int(bitstring.sum().detach().cpu().item())

        # MDS 是最小化问题：
        # 先保证未支配节点少，再保证选中节点少。
        score = undominated_num * (n + 1) + selected_num

        if score < best_score:
            best_score = score
            best_bitstring = bitstring.clone().detach()

        if epoch % 1000 == 0:
            print(
                f"Epoch: {epoch}, "
                f"Loss: {loss_}, "
                f"Obj: {loss_obj.detach().item()}, "
                f"Cons: {loss_cons.detach().item()}, "
                f"Gini: {loss_gini.detach().item()}, "
                f"GiniCoef: {gini_coef}, "
                f"Undominated: {undominated_num}, "
                f"Selected: {selected_num}"
            )

        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f"Stopping early on epoch {epoch} (patience: {patience})")
            break

        # update loss tracking
        prev_loss = loss_

        # run optimization with backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_gnn = time() - t_gnn_start
    print(f"GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}")
    print(f"GNN final continuous loss: {loss_}")
    print(f"GNN best discrete score: {best_score}")

    final_bitstring = (probs.detach() >= prob_threshold) * 1
    final_bitstring = final_bitstring.type(B.dtype).to(B.device)

    return net, epoch, final_bitstring, best_bitstring


# =========================
# 9. 单次运行 PI-GNN
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

    # 3. 构造 MDS 闭邻域矩阵
    B = build_closed_neighborhood_matrix(nx_graph)

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
    print("Running PI-GNN for MDS...")
    start_time = time()

    _, epoch, final_bitstring, best_bitstring = run_gnn_training_mds(
        B,
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
    raw_mds_size, raw_dom_set, raw_undominated_nodes, raw_is_valid, raw_coverage_ratio = postprocess_gnn_mds(
        best_bitstring,
        nx_graph,
    )

    if REPAIR:
        final_solution, added_nodes = repair_gnn_mds(best_bitstring, nx_graph)
    else:
        final_solution = best_bitstring
        added_nodes = []

    size_mds, dom_set, undominated_nodes, is_valid, coverage_ratio = postprocess_gnn_mds(
        final_solution,
        nx_graph,
    )

    # 7. 计算 final loss，失败时不影响主流程
    try:
        inputs = embed.weight
        probs = net(graph_dgl, inputs)[:, 0]
        final_loss, final_obj, final_cons, final_gini, _, _ = loss_func_mds(
            probs,
            B,
            epoch + 1,
        )
        final_loss_value = float(final_loss.detach().cpu().item())
        final_obj_value = float(final_obj.detach().cpu().item())
        final_cons_value = float(final_cons.detach().cpu().item())
        final_gini_value = float(final_gini.detach().cpu().item())
    except Exception:
        final_loss_value = None
        final_obj_value = None
        final_cons_value = None
        final_gini_value = None

    print(
        f"PI-GNN result | "
        f"MDS size: {size_mds} | "
        f"Undominated nodes: {len(undominated_nodes)} | "
        f"Valid: {is_valid} | "
        f"Raw MDS size: {raw_mds_size} | "
        f"Raw undominated: {len(raw_undominated_nodes)} | "
        f"Repair added: {len(added_nodes)} | "
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

        "mds_size": int(size_mds),
        "undominated_nodes": int(len(undominated_nodes)),
        "is_valid": bool(is_valid),
        "coverage_ratio": float(coverage_ratio),
        "repair_added_nodes": int(len(added_nodes)),

        "raw_mds_size": int(raw_mds_size),
        "raw_undominated_nodes": int(len(raw_undominated_nodes)),
        "raw_is_valid": bool(raw_is_valid),
        "raw_coverage_ratio": float(raw_coverage_ratio),

        "epoch": int(epoch),
        "time_sec": float(train_time),
        "final_loss": final_loss_value,
        "final_obj_loss": final_obj_value,
        "final_cons_loss": final_cons_value,
        "final_gini_loss": final_gini_value,
    }

    # 释放显存
    del net, embed, optimizer, graph_dgl, B

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =========================
# 10. 汇总单个数据集的 10 次结果
# =========================

def summarize_dataset_results(dataset: Datasets, dataset_results):
    """
    汇总一个数据集 10 次运行结果。

    best_mds_size：
        优先在合法解中选最小 MDS；
        如果 10 次都不合法，则从所有结果中选未支配节点最少、mds_size 最小的结果。

    avg_mds_size：
        10 次运行的平均 MDS 大小。
    """
    df = pd.DataFrame(dataset_results)

    # 去掉失败的运行
    df = df[df["mds_size"].notna()].copy()

    if len(df) == 0:
        return {
            "dataset_enum": dataset.name,
            "dataset_value": dataset.value,
            "num_nodes": None,
            "num_edges": None,
            "best_mds_size": None,
            "best_run": None,
            "best_seed": None,
            "best_undominated_nodes": None,
            "best_is_valid": False,
            "avg_mds_size": None,
            "std_mds_size": None,
            "avg_valid_mds_size": None,
            "std_valid_mds_size": None,
            "valid_runs": 0,
            "avg_undominated_nodes": None,
            "avg_time_sec": None,
            "std_time_sec": None,
            "best_time_sec": None,
        }

    valid_df = df[df["is_valid"] == True]

    if len(valid_df) > 0:
        best_row = valid_df.loc[valid_df["mds_size"].idxmin()]
    else:
        df["score"] = df["undominated_nodes"] * (df["num_nodes"] + 1) + df["mds_size"]
        best_row = df.loc[df["score"].idxmin()]

    if len(valid_df) > 0:
        avg_valid_mds_size = float(valid_df["mds_size"].mean())
        std_valid_mds_size = float(valid_df["mds_size"].std(ddof=0))
    else:
        avg_valid_mds_size = None
        std_valid_mds_size = None

    summary = {
        "dataset_enum": dataset.name,
        "dataset_value": dataset.value,
        "num_nodes": int(df["num_nodes"].iloc[0]),
        "num_edges": int(df["num_edges"].iloc[0]),

        # 10 次中最好结果
        "best_mds_size": int(best_row["mds_size"]),
        "best_run": int(best_row["run"]),
        "best_seed": int(best_row["seed"]),
        "best_undominated_nodes": int(best_row["undominated_nodes"]),
        "best_is_valid": bool(best_row["is_valid"]),
        "best_repair_added_nodes": int(best_row["repair_added_nodes"]),

        # 所有运行的平均结果
        "avg_mds_size": float(df["mds_size"].mean()),
        "std_mds_size": float(df["mds_size"].std(ddof=0)),

        # 只统计合法运行的平均结果
        "avg_valid_mds_size": avg_valid_mds_size,
        "std_valid_mds_size": std_valid_mds_size,

        # 原始输出结果
        "avg_raw_mds_size": float(df["raw_mds_size"].mean()),
        "avg_raw_undominated_nodes": float(df["raw_undominated_nodes"].mean()),
        "avg_repair_added_nodes": float(df["repair_added_nodes"].mean()),

        # 合法性统计
        "valid_runs": int(df["is_valid"].sum()),
        "avg_undominated_nodes": float(df["undominated_nodes"].mean()),

        # 时间统计
        "avg_time_sec": float(df["time_sec"].mean()),
        "std_time_sec": float(df["time_sec"].std(ddof=0)),
        "best_time_sec": float(best_row["time_sec"]),
    }

    return summary


# =========================
# 11. 主函数：一次运行得到全部结果
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

                    "mds_size": None,
                    "undominated_nodes": None,
                    "is_valid": False,
                    "coverage_ratio": None,
                    "repair_added_nodes": None,

                    "raw_mds_size": None,
                    "raw_undominated_nodes": None,
                    "raw_is_valid": False,
                    "raw_coverage_ratio": None,

                    "epoch": None,
                    "time_sec": None,
                    "final_loss": None,
                    "final_obj_loss": None,
                    "final_cons_loss": None,
                    "final_gini_loss": None,
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
            f"Best MDS size: {summary['best_mds_size']} | "
            f"Average MDS size: {summary['avg_mds_size']} ± {summary['std_mds_size']} | "
            f"Valid runs: {summary['valid_runs']}/{NUM_RUNS} | "
            f"Average time: {summary['avg_time_sec']}s"
        )
        print("-" * 100)

    # =========================
    # 12. 保存结果
    # =========================

    current_path = Path(__file__).parent
    save_dir = current_path / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    detail_path = save_dir / "mds_pignn_run_details.csv"
    summary_path = save_dir / "mds_pignn_summary.csv"

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

        "mds_size",
        "undominated_nodes",
        "is_valid",
        "coverage_ratio",
        "repair_added_nodes",

        "raw_mds_size",
        "raw_undominated_nodes",
        "raw_is_valid",
        "raw_coverage_ratio",

        "epoch",
        "time_sec",
        "final_loss",
        "final_obj_loss",
        "final_cons_loss",
        "final_gini_loss",
        "error",
    ]
    detail_columns = [c for c in detail_columns if c in df_detail.columns]
    df_detail = df_detail[detail_columns]

    summary_columns = [
        "dataset_enum",
        "dataset_value",
        "num_nodes",
        "num_edges",

        "best_mds_size",
        "best_run",
        "best_seed",
        "best_undominated_nodes",
        "best_is_valid",
        "best_repair_added_nodes",

        "avg_mds_size",
        "std_mds_size",
        "avg_valid_mds_size",
        "std_valid_mds_size",

        "avg_raw_mds_size",
        "avg_raw_undominated_nodes",
        "avg_repair_added_nodes",

        "valid_runs",
        "avg_undominated_nodes",

        "avg_time_sec",
        "std_time_sec",
        "best_time_sec",
    ]
    summary_columns = [c for c in summary_columns if c in df_summary.columns]
    df_summary = df_summary[summary_columns]

    df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("全部 PI-GNN-MDS 实验完成。")
    print(f"每次运行详细结果已保存到：{detail_path}")
    print(f"每个数据集汇总结果已保存到：{summary_path}")
    print("=" * 100)

    print("\n汇总结果：")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()