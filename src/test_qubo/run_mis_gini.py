import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.mis.model import MISNet
from src.mis.loss_gini import loss_mis_gini_qubo
from src.mis.utils import is_maximal_independent_set

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

    data_path = Datasets.Graph_Pubmed.path
    graph = from_file_to_graph(data_path, True).to(get_device())

    # v = 100
    # p = 0.1
    # e = int(p * v * (v - 1) / 2)

    # graph = generate_data("graph", v=v, e=e).to(get_device())

    init_feature_dim = 512
    x = torch.rand((graph.num_v, init_feature_dim), device=device)

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 512, hidden_channels=512, num_layers=2, jk="cat", drop_rate=0)]
    # shared_layers = [TransformerEncoderLayer(1024, 16, 1024)]
    shared_layers = []

    obj_layers = [Layer(LayerType.LINEAR, 512, 256, use_bn=False, drop_rate=0.1),
                  Layer(LayerType.LINEAR, 256, 1, use_bn=False, drop_rate=0.1)]

    net = MISNet(gnn_layers, obj_layers).to(device)

    gini_cons_lambda = lambda e, n: (-18000 + e) / 500   

    loss, outs = run_qubo(
        "mis",
        net,
        x,
        graph,  
        20000,
        loss_mis_gini_qubo,
        1e-4,
        gini_cof_lambda=gini_cons_lambda
    )

    # solution = (outs[:, 0] > 0.5).float()
    # is_maximal = is_maximal_independent_set(solution, graph)
    # print("Is maximal independent set:", is_maximal)

    # x = outs[:, 0].detach().cpu().view(-1)

    # print("num > 0.7:", (x > 0.7).sum().item())
    # print("num in (0.4, 0.6):", ((x > 0.4) & (x < 0.6)).sum().item())
    # print("num in (0.1, 0.9):", ((x > 0.1) & (x < 0.9)).sum().item())
    # print("min:", x.min().item(), "max:", x.max().item(), "mean:", x.mean().item())
