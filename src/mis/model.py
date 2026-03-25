import torch
import torch.nn as nn
from ..core import Layer
from typing import List


class Net(nn.Module):
    def __init__(self, gnn_layers: List[Layer], obj_layers: List[Layer]):
        super(Net, self).__init__()
        self.gnn_layers = nn.ModuleList(gnn_layers)  # 用于图神经网络的层
        self.obj_layers = nn.ModuleList(obj_layers)  # 用于目标输出的层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, **kwargs):
        # 处理GNN层
        for layer in self.gnn_layers:
            x = layer(x, graph, edge_index, **kwargs)
        
        # 处理输出层
        for layer in self.obj_layers:
            x = layer(x, graph, edge_index, **kwargs)
        
        x = self.sigmoid(x)

        return x