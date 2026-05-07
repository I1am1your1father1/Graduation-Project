import dgl
import torch
import random
import os
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from pathlib import Path
from collections import OrderedDict, defaultdict
from dgl.nn.pytorch import GraphConv
from itertools import chain, islice, combinations
from time import time

# MacOS can have issues with MKL. For more details, see
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# fix seed to ensure consistent results
seed_value = 1
random.seed(seed_value)        # seed python RNG
np.random.seed(seed_value)     # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

from utils import generate_graph, get_gnn, run_gnn_training, qubo_dict_to_torch, gen_combinations, loss_func


# =========================
# 1. 你的数据集枚举类
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
        data_path = os.path.join(current_path, "..", "data", type_path, f"{self.value}.txt")
        return data_path

    @property
    def type(self):
        return self.value.split("_")[-1]


# =========================
# 2. 读取你的真实图数据集
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
        2. 原始节点编号可能很大，例如 103482，所以必须重新编号为 0,1,...,n-1；
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

    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)

    # 保持和原 generate_graph 函数类似的 OrderedGraph 形式
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)

    print(
        f"Loaded graph with n={nx_graph.number_of_nodes()}, "
        f"m={nx_graph.number_of_edges()}"
    )

    return nx_graph


# helper function to generate Q matrix for Maximum Independent Set problem (MIS)
def gen_q_dict_mis(nx_G, penalty=2):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.

    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty

    # all diagonal terms get -1
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -1

    return Q_dic


# Calculate results given bitstring and graph definition, includes check for violations
def postprocess_gnn_mis(best_bitstring, nx_graph):
    """
    helper function to postprocess MIS results

    Input:
        best_bitstring: bitstring as torch tensor
    Output:
        size_mis: Size of MIS (int)
        ind_set: MIS (list of integers)
        number_violations: number of violations of ind.set condition
    """

    # get bitstring as list
    bitstring_list = list(best_bitstring)

    # compute cost
    size_mis = sum(bitstring_list)

    # get independent set
    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))

    print('Calculating violations...')
    # check for violations
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return size_mis, ind_set, number_violations


# =========================
# 3. 图参数
# =========================

# Graph hypers
n = 100
d = 3
p = None
graph_type = 'reg'

# NN learning hypers
number_epochs = int(1e5)
learning_rate = 1e-4
PROB_THRESHOLD = 0.5

# Early stopping to allow NN to train to near-completion
tol = 1e-4          # loss must change by more than tol, or trigger
patience = 100      # number early stopping triggers before breaking loop


# =========================
# 4. 选择你的真实数据集
# =========================

# dataset = Datasets.Graph_Cora
# dataset = Datasets.Graph_Citeseer
# dataset = Datasets.Graph_Amazon_PC
# dataset = Datasets.Graph_Amazon_Photo
# dataset = Datasets.Graph_Pubmed
# dataset = Datasets.Graph_dblp
# dataset = Datasets.Graph_bat
# dataset = Datasets.Graph_eat
dataset = Datasets.Graph_uat


# =========================
# 5. 构造图
# =========================

# 原始随机图生成代码保留，以后如果还想用随机图，取消注释即可
# nx_graph = generate_graph(n=n, d=d, p=p, graph_type=graph_type, random_seed=seed_value)

# 使用你的真实数据集
nx_graph = load_graph_from_dataset(dataset)

# 更新真实图的节点数量
n = nx_graph.number_of_nodes()

# get DGL graph from networkx graph, load onto device
graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
graph_dgl = dgl.add_self_loop(graph_dgl)
graph_dgl = graph_dgl.to(TORCH_DEVICE)


# =========================
# 6. 构造 QUBO 矩阵
# =========================

# Construct Q matrix for graph
q_torch = qubo_dict_to_torch(
    nx_graph,
    gen_q_dict_mis(nx_graph),
    torch_dtype=TORCH_DTYPE,
    torch_device=TORCH_DEVICE
)


# =========================
# 7. 可视化原始图
# =========================

# # 如果图很大，画图会很慢，可以注释掉这两行
# pos = nx.kamada_kawai_layout(nx_graph)
# nx.draw(nx_graph, pos, with_labels=True, node_color=[[.7, .7, .7]])


# =========================
# 8. GNN 超参数
# =========================

# Establish dim_embedding and hidden_dim values
dim_embedding = int(np.sqrt(n))
hidden_dim = int(dim_embedding / 2)

# 防止特别小的图导致 hidden_dim 为 0
if dim_embedding < 1:
    dim_embedding = 1
if hidden_dim < 1:
    hidden_dim = 1

# Establish pytorch GNN + optimizer
opt_params = {'lr': learning_rate}
gnn_hypers = {
    'dim_embedding': dim_embedding,
    'hidden_dim': hidden_dim,
    'dropout': 0.0,
    'number_classes': 1,
    'prob_threshold': PROB_THRESHOLD,
    'number_epochs': number_epochs,
    'tolerance': tol,
    'patience': patience
}

net, embed, optimizer = get_gnn(n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)

# For tracking hyperparameters in results object
gnn_hypers.update(opt_params)


# =========================
# 9. 运行 GNN
# =========================

print('Running GNN...')
gnn_start = time()

_, epoch, final_bitstring, best_bitstring = run_gnn_training(
    q_torch,
    graph_dgl,
    net,
    embed,
    optimizer,
    gnn_hypers['number_epochs'],
    gnn_hypers['tolerance'],
    gnn_hypers['patience'],
    gnn_hypers['prob_threshold']
)

gnn_time = time() - gnn_start

final_loss = loss_func(final_bitstring.float(), q_torch)
final_bitstring_str = ','.join([str(x) for x in final_bitstring])


# =========================
# 10. 处理 GNN 结果
# =========================

# Process bitstring reported by GNN
size_mis, ind_set, number_violations = postprocess_gnn_mis(best_bitstring, nx_graph)
gnn_tot_time = time() - gnn_start

print(f'Independence number found by GNN is {size_mis} with {number_violations} violations')
print(f'Took {round(gnn_tot_time, 3)}s, model training took {round(gnn_time, 3)}s')


# # =========================
# # 11. 可视化 GNN 求解结果
# # =========================

# # 如果图很大，画图会很慢，可以注释掉这一部分
# # Note no light-blue nodes are connected by an edge
# color_map = ['orange' if (best_bitstring[node] == 0) else 'lightblue' for node in nx_graph.nodes]
# nx.draw(nx_graph, pos, with_labels=True, node_color=color_map)