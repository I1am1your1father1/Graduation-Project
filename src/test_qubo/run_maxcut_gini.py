import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.max_cut.model import MAXCUTNet
from src.max_cut.loss_gini import loss_maxcut_gini_qubo

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
    init_feature_dim = 128

    data_path = Datasets.Graph_Pubmed.path
    graph = from_file_to_graph(data_path, True).to(get_device())

    # v = 100
    # p = 0.1
    # e = int(p * v * (v - 1) / 2)

    # graph = generate_data("graph", v=v, e=e).to(get_device())

    init_feature_dim = 512
    x = torch.rand((graph.num_v, init_feature_dim), device=device)

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 512, hidden_channels=512, num_layers=2, jk="cat", drop_rate=0)]

    shared_layers = []

    obj_layers = [Layer(LayerType.LINEAR, 512, 256, use_bn=False, drop_rate=0.1),
                  Layer(LayerType.LINEAR, 256, 1, use_bn=False, drop_rate=0.1)]
    
    net = MAXCUTNet(gnn_layers + shared_layers + obj_layers).to(device)

    gini_cons_lambda = lambda e, n: (-15000 + e) / 1000

    loss, outs, eval_result = run_qubo(
        "max_cut",
        net,
        x,
        graph,
        20000,
        loss_maxcut_gini_qubo,
        1e-4,
        evaluate=True,
        gini_cof_lambda=gini_cons_lambda,
    )

    print(eval_result)
