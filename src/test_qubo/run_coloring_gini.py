import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.coloring.model import DualHeadNet
from src.coloring.loss_gini import loss_coloring_gini_qubo

from src.core import init, get_device, run_qubo, Layer, LayerType, Datasets
from src.utils import from_file_to_graph, generate_data


def count_values_in_range(outs: torch.Tensor) -> int:
    mask = (outs > 0.01) & (outs < 0.99)
    count = torch.sum(mask).item()
    return count

if __name__ == "__main__":
    # Find Best sol Cora: 7 22s
    init(cuda_index=1, reproducibility=False)
    device = get_device()

    data_path = Datasets.Graph_Cora.path
    graph = from_file_to_graph(data_path, True).to(get_device())

    # v = 100
    # p = 0.1
    # e = int(p * v * (v - 1) / 2)

    # graph = generate_data("graph", v=v, e=e).to(get_device())

    init_feature_dim = 128
    x = torch.rand((graph.num_v, init_feature_dim), device=device)

    K = int(graph.A.to_dense().sum(dim=1).max().item()) + 1

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 128, hidden_channels=128, num_layers=2, jk="cat", drop_rate=0)]
    # shared_layers = [TransformerEncoderLayer(1024, 16, 1024)]
    shared_layers = []

    cons_layers = [Layer(LayerType.LINEAR, 128, 64, use_bn=False, drop_rate=0.1),
                  Layer(LayerType.LINEAR, 64, K, use_bn=False, drop_rate=0.1)]

    obj_layers = [Layer(LayerType.LINEAR, 128, 64, use_bn=False, drop_rate=0.1),
                  Layer(LayerType.LINEAR, 64, K, use_bn=False, drop_rate=0.1)]

    net = DualHeadNet(gnn_layers, shared_layers, cons_layers, obj_layers).to(device)

    gini_cons_lambda = lambda e, n: (-1000 + e) / 1000

    loss, outs = run_qubo(
        "coloring",
        net,
        x,
        graph,  
        5000,
        loss_coloring_gini_qubo,
        1e-4,
        gini_cons_cof_lambda=gini_cons_lambda,
        cons_cof_lambda=lambda e, n: 3.0, # 3.0
        obj_cof_lambda=lambda e, n: 1.5, # 1.0
        obj_cons_cof_lambda=lambda e, n: 0.5 # 2.0
    )

outs_cons, outs_obj = outs

print("final loss:", loss)
print("outs_cons shape:", outs_cons.shape)
print("outs_obj shape:", outs_obj.shape)
print("used color score:", outs_obj)
print("num soft values in cons:", count_values_in_range(outs_cons))