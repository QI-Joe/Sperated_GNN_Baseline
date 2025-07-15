from utils.my_dataloader import Temporal_Dataloader, NodeIdxMatching, Dynamic_Dataloader, data_load
import numpy as np
from numpy import ndarray
import copy
from typing import List, Tuple, Optional, Any
import torch

class Imbanlance(object): 
    def __init__(self, ratio: float, train_ratio: float, *args, **kwargs):
        super(Imbanlance, self).__init__(*args, **kwargs)
        self.imbalance_ratio = ratio
        self.train_ratio = train_ratio
        self.seen_node: ndarray = None
        self.node_match_list: Optional[NodeIdxMatching | Any] = None
        
    def __call__(self, data: Temporal_Dataloader, *args, **kwds):
        """
        Imbalance Data Evaluation. In the node classification task, given a dataset G = (𝐺𝑖 , 𝑦𝑖 ), we simulate class imbalance by setting \n
        the proportions of training samples per class as {1, 1/2^𝛽 , 1/3^𝛽 , . . . , 1/|Y|^𝛽 }, where 𝛽 ∈ {0, 0.5, 1, 1.5, 2} controls the imbalance ratio. \n
        The num-ber of samples in the first class is fixed under all 𝛽 values.
        """
        nodes, label = data.my_n_id.node["index"].values, data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else data.y
        node_num, node_match_list = data.num_nodes, copy.deepcopy(data.my_n_id.node)
        seen_node, seen_node_label = nodes[:int(node_num*self.train_ratio)], label[:int(node_num*self.train_ratio)]
        
        # All for implmentation
        # available_node_list: list[np.ndarray] = self.pooling_check(seen_node, node_match_list)
        
        uniqclass, uniquenum = np.unique(seen_node_label, return_counts=True)
        fixed_sample = uniquenum[0]
        sample_per_classes = [int(fixed_sample/((i+1)**self.imbalance_ratio)) for i in range(len(uniqclass))]
        
        selected_idx, outside_select = [], []
        for class_label, num_samples in zip(uniqclass, sample_per_classes):
            class_idx = np.where(seen_node_label == class_label)[0]
            if num_samples>0 and len(class_idx)>0:
                selected = np.random.choice(class_idx, min(num_samples, len(class_idx)), replace=False)
                not_selected = list(set(class_idx.tolist()) - set(selected.tolist()))
                selected_idx.extend(selected)
                outside_select.extend(not_selected)
        
        """
        Attention! There should be a assert to evaluate one thing:
        (np.array(selected_idx.extend(outside_select)) == seen_node).all() == True
        """
        
        train_mask, val_mask, nn_val_mask = np.zeros(node_num, dtype=bool), np.zeros(node_num, dtype=bool), np.zeros(node_num, dtype=bool)
        
        outside_select.extend(nodes[int(node_num*self.train_ratio):].tolist())
        nn_val_node_idx = np.array(outside_select)
        train_mask[np.array(selected_idx)], val_mask[int(node_num*self.train_ratio):], nn_val_mask[nn_val_node_idx] = True, True, True
        
        data.train_val_mask_injection(train_mask, val_mask, nn_val_mask)
        
        self.seen_node = node_match_list.values[selected_idx, 1]
        
        return data
    
    def test_processing(self, t1_data: Temporal_Dataloader):
        t1_nodenum = t1_data.num_nodes
        t1_match_list: ndarray = t1_data.my_n_id.node.values # "index", "original node idx", "label"
        
        # seen_node = self.seen_node
        # t_match_list: ndarray = self.node_match_list.node.values # "index", "original node idx", "label"
        
        t1_unseen_node_mask = ~np.isin(t1_match_list[:, 1], self.seen_node)
        nn_test_mask = t1_unseen_node_mask
        
        t1_data.test_mask_injection(nn_test_mask)
        
        return t1_data
        
class Few_Shot_Learning(object):
    def __init__(self, fsl_num: int, *args, **kwargs):
        super(Few_Shot_Learning, self).__init__(*args, **kwargs)
        self.fsl_num = fsl_num
        self.seen_node: ndarray = None
        
    def __call__(self, data: Temporal_Dataloader, *args, **kwds):
        """
        Few-shot Evaluation. Specifically, For graph classification \n
        tasks, given a training graph dataset G = {(𝐺𝑖 , 𝑦𝑖 )}, we set the \n
        number of training graphs per class as 𝛾 ∈ {10, 20, 30, 40, 50}. 
        """
        node_num, label = data.num_nodes, data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else data.y
        uniquclss, uniqunum = np.unique(label, return_counts=True)
        training_data, node_match_list = [], data.my_n_id.node.values # "index", "original node idx", "label"
        
        for cls in uniquclss:
            class_indices = np.where(label == cls)[0]
            np.random.shuffle(class_indices)
            
            num_samples = min(self.fsl_num, len(class_indices))
            current_cls_selelcted_indices = class_indices[: num_samples]
            training_data.extend(current_cls_selelcted_indices)
            
        self.seen_node = node_match_list[training_data, 1] # training data could direct retrieve on match list index to locate original node idx
        train_mask = np.zeros(node_num, dtype=bool)
        train_mask[training_data] = True
        val_mask, nn_val_mask = copy.deepcopy(~train_mask), copy.deepcopy(~train_mask) # Attention here, nn_val_mask will equal to val_mask
        
        data.train_val_mask_injection(train_mask, val_mask, nn_val_mask)
        return data
    
    def test_processing(self, t1_data: Temporal_Dataloader):
        t1_nodenum = t1_data.num_nodes
        t1_match_list: ndarray = t1_data.my_n_id.node.values # "index", "original node idx", "label"
        
        t1_unseen_node_mask = ~np.isin(t1_match_list[:, 1], self.seen_node)
        nn_test_mask = t1_unseen_node_mask
        
        t1_data.test_mask_injection(nn_test_mask)
        
        return t1_data