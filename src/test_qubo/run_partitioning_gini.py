import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.partitioning.model import PARTITIONNet
from src.partitioning.loss_gini import loss_partitioning_gini_qubo

from src.core import init, get_device, run_qubo, Layer, LayerType, Datasets
from src.utils import from_file_to_graph, generate_data


def count_values_in_range(outs: torch.Tensor) -> int:
    mask = (outs > 0.01) & (outs < 0.99)
    count = torch.sum(mask).item()
    return count

if __name__ == "__main__":
    # Find Best sol Cora: 7 22s
    init(cuda_index=1, reproducibility=True)
    device = get_device()

    # data_path = Datasets.Graph_Cora.path
    # graph = from_file_to_graph(data_path, True).to(get_device())

    v = 100
    p = 0.1
    e = int(p * v * (v - 1) / 2)

    graph = generate_data("graph", v=v, e=e).to(get_device())

    init_feature_dim = 128
    x = torch.rand((graph.num_v, init_feature_dim), device=device)

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 128, hidden_channels=128, num_layers=2, jk="cat", drop_rate=0)]
    # shared_layers = [TransformerEncoderLayer(1024, 16, 1024)]
    shared_layers = []

    obj_layers = [Layer(LayerType.LINEAR, 128, 64, use_bn=False, drop_rate=0.1),
                  Layer(LayerType.LINEAR, 64, 2, use_bn=False, drop_rate=0.1)]

    layers = gnn_layers + shared_layers + obj_layers

    net = PARTITIONNet(layers).to(device)

    gini_cons_lambda = lambda e, n: (e - 1500) / 1000

    loss, outs = run_qubo(
        "partitioning",
        net,
        x,
        graph,  
        3000,
        loss_partitioning_gini_qubo,
        1e-4,
        gini_cof_lambda=gini_cons_lambda,
    )
