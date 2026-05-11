import torch
import torch.nn as nn
from ..core import Layer
from typing import List


class MISNet(nn.Module):
    def __init__(self, gnn_layers: List[Layer], obj_layers: List[Layer]):
        super(MISNet, self).__init__()
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.obj_layers = nn.ModuleList(obj_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, **kwargs):
        
        for layer in self.gnn_layers:
            x = layer(x, graph, edge_index, **kwargs)

        for layer in self.obj_layers:
            x = layer(x, graph, edge_index, **kwargs)
        
        x = self.sigmoid(x)

        return x