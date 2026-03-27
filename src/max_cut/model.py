from .. import Layer
from typing import List

import torch
import torch.nn as nn


class MAXCUTNet(nn.Module):

    def __init__(self, layers: List[Layer]):
        super(MAXCUTNet, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, edge_weight=None, **kwargs):
        for layer in self.layers:
            x = layer(x, graph, edge_index, edge_weight, **kwargs)
        x = self.softmax(x)
        return x
