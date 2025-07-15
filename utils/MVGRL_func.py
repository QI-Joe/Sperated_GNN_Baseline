import torch
from torch_geometric.data import Data
import random
from typing import Union, Optional, Any
from utils.robustness_injection import Imbanlance, Few_Shot_Learning

def get_split(num_samples: int, emb:Union[torch.Tensor | tuple[torch.Tensor]], data: Union[Data | tuple[Data, Data]], \
              Nontemproal: bool=False, transfer: Optional[Imbanlance | Any] = None, \
              train_ratio: float = 0.1, test_ratio: float = 0.8):
    random.seed(2024)
    torch.manual_seed(2024)
    
    nn_valid, nn_test = None, None
    if Nontemproal:
        assert train_ratio + test_ratio < 1
        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * train_ratio* 2)
        indices = torch.randperm(num_samples)

        assert not isinstance(data, list), f"expected to be static graph, but got dynamic-fit graph with {len(data)} graphs"
        assert not isinstance(emb, list), f"expected to be static graph, but got dynamic-fit graph with {len(emb)} embeddings"

        graph, emb = data, emb
        train = {
            "emb": emb[indices[:train_size]],
            "label": graph.y[indices[:train_size]],
        }
        valid = {
            "emb": emb[indices[train_size: val_size]],
            "label": graph.y[indices[train_size: val_size]],
        }
        test = {
            "emb": emb[indices[val_size:]],
            "label": graph.y[indices[val_size:]],
        }
    else:
        train_size = int(num_samples * train_ratio)
        indices = torch.randperm(num_samples)

        graph1, graph2 = data
        emb1, emb2 = emb
        
        train = {
            "emb": emb1[indices[:train_size]],
            "label": graph1.y[indices[:train_size]],
        }
        valid = {
            "emb": emb1[indices[train_size: ]],
            "label": graph1.y[indices[train_size:]],
        }
        test = {
            "emb": emb2,
            "label": graph2.y,
        }
        
        if transfer != None:
            if isinstance(transfer, Imbanlance):
                graph1 = transfer(graph1)
                graph2 = transfer.test_processing(graph2)
                train = {
                        "emb": emb1[graph1.train_mask],
                        "label": graph1.y[graph1.train_mask],
                }
                valid = {
                    "emb": emb1[graph1.val_mask],
                    "label": graph1.y[graph1.val_mask],
                }                
                nn_valid = {
                    "emb": emb1[graph1.nn_val_mask], 
                    "label": graph1.y[graph1.nn_val_mask]
                }
                nn_test = {
                    "emb": emb2[graph2.nn_test_mask],
                    "label": graph2.y[graph2.nn_test_mask]
                }
            elif isinstance(transfer, Few_Shot_Learning):
                graph1 = transfer(graph1)
                graph2 = transfer.test_processing(graph2)
                train = {
                    "emb": emb1[graph1.train_mask],
                    "label": graph1.y[graph1.train_mask],
                }
                valid = {
                    "emb": emb1[graph1.val_mask],
                    "label": graph1.y[graph1.val_mask],
                }       
                nn_valid = {
                    "emb": emb1[graph1.nn_val_mask], 
                    "label": graph1.y[graph1.nn_val_mask]
                } # nn_valid setting in FSL will equal to valid, put it here is for comptability with Imbalance         
                nn_test = {
                    "emb": emb2[graph2.nn_test_mask],
                    "label": graph2.y[graph2.nn_test_mask]
                }
    return (train, valid, test, nn_valid, nn_test)
