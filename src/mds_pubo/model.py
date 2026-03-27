import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import torch
import torch.nn as nn
from core import Layer
from typing import List


class MDSNet(nn.Module):
    def __init__(self, layers: List[Layer]):
        super(MDSNet, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, graph=None, edge_index=None, **kwargs):
        for layer in self.layers:
            x = layer(x, graph, edge_index, **kwargs)
        x = self.softmax(x)

        return x
