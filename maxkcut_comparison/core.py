import os
from enum import Enum
from pathlib import Path


class Datasets(Enum):
    Graph_Cora = "cora_graph"
    Graph_Citeseer = "citeseer_graph"
    Graph_Amazon_PC = "amazon_electronics_computers_graph"
    Graph_Amazon_Photo = "amazon_electronics_photo_graph"
    Graph_Pubmed = "pubmed_graph"
    Graph_dblp = "dblp_graph"
    Graph_bat = "bat_graph"
    Graph_eat = "eat_graph"
    Graph_uat = "uat_graph"

    @property
    def path(self):
        current_path = Path(__file__).parent
        type_path = self.value.split("_")[-1]
        data_path = os.path.join(current_path, "..", "data", type_path, f"{self.value}.txt")
        return data_path
    
    @property
    def type(self):
        return self.value.split("_")[-1]