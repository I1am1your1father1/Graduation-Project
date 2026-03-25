import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from src.mds_pubo.model import Net
from src.mds_pubo.loss_gini import loss_mds_gini_pubo

from src.core import init, get_device, run_graph_pubo, Layer, LayerType, Datasets
from src.utils import from_file_to_graph, generate_data


if __name__ == "__main__":
    # Find Best sol Cora: 7 22s
    init(cuda_index=1, reproducibility=True)
    init_feature_dim = 128

    #data_path = Datasets.Graph_bat.path
    # graph = from_file_to_graph("/homebs/wmm/mcds-master/mcds-master/data/graph/tt1.txt", True).to(get_device())

    v = 100
    p = 0.1
    e = int(p * v * (v - 1) / 2)

    graph = generate_data("graph", v=v, e=e).to(get_device())

    x = torch.rand((graph.num_v, init_feature_dim))

    gnn_layers = [Layer(LayerType.GRAPHSAGE, init_feature_dim, 128, hidden_channels=128, num_layers=2, jk="cat", drop_rate=0)]

    shared_layers = []

    obj_layers = [Layer(LayerType.LINEAR, 128, 64, use_bn=False, drop_rate=0.1),
                  Layer(LayerType.LINEAR, 64, 2, use_bn=False, drop_rate=0.1)]
  
    all_layers = gnn_layers + obj_layers
    net = Net(all_layers)

    loss, outs, _ = run_graph_pubo(
        "mds",
        net,
        x,
        graph,  
        2000,
        loss_mds_gini_pubo,
        1e-4,
    )

def count_values_in_range(outs: torch.Tensor) -> int:
    mask = (outs > 0.01) & (outs < 0.99)
    count = torch.sum(mask).item()
    return count
    
print("count values in range:", count_values_in_range(outs))
print(1-outs)
