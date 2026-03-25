from typing import List, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool


class DualHeadNet(nn.Module):
    def __init__(
        self,
        gnn_layers,
        shared_layers,
        cons_layers,
        obj_layers,
    ):
        super().__init__()
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.shared_layers = nn.ModuleList(shared_layers)
        self.cons_layers = nn.ModuleList(cons_layers)
        self.obj_layers = nn.ModuleList(obj_layers)
        self.act = nn.ModuleDict({
            "softmax": nn.Softmax(dim=1), 
            "sigmoid": nn.Sigmoid()
        })

    def forward(self, x: torch.Tensor, graph: object, edge_index, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.gnn_layers:
            x = layer(x, graph, edge_index)
        for layer in self.shared_layers:
            x = layer(x)
        cons = x
        if len(self.cons_layers) != 0:
            for layer in self.cons_layers:
                cons = layer(cons)

        obj = x
        if len(self.obj_layers) != 0:
            for layer in self.obj_layers:
                obj = layer(obj)

        cons = self.act["softmax"](cons)

        # batch = torch.zeros(obj.shape[0], dtype=torch.long, device=obj.device)
        # obj = global_max_pool(self.softmax(obj), batch)
        
        batch = torch.zeros(obj.shape[0], dtype=torch.long, device=obj.device)
        obj = global_max_pool(cons, batch)

        return (cons, obj)
