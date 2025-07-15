from utils.data_processing import get_link_prediction_data
import numpy as np
import time
import sys
import os
import torch
import torch.nn as nn
from utils.utils import convert2pyg_batch_data

dataset_name = "askubuntu"
SNAPSHOT = 10
val_ratio = 0.2
test_ratio = 1.0

node_raw_features, edge_raw_features, data_list = \
        get_link_prediction_data(dataset_name=dataset_name, snapshot=SNAPSHOT, val_ratio=val_ratio, test_ratio=test_ratio)
        
print(f"Dataset: {dataset_name}, Snapshots: {SNAPSHOT}, Val Ratio: {val_ratio}, Test Ratio: {test_ratio}")
views = len(data_list)

for idx, data in enumerate(data_list):
    full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = data_list[idx]
    pyg_train, pyg_val, pyg_test, pyg_new_node_val, pyg_new_node_test = convert2pyg_batch_data([train_data, val_data, test_data, new_node_val_data, new_node_test_data], node_raw_features, edge_raw_features)
    
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "new_node_val": new_node_val_data,
        "new_node_test": new_node_test_data
    }
    pyg_datasets = {
        "train": pyg_train,
        "val": pyg_val,
        "test": pyg_test,
        "new_node_val": pyg_new_node_val,
        "new_node_test": pyg_new_node_test
    }

    print(f"\nSnapshot {idx+1}/{views}")
    print(f"Full data: num_nodes={full_data.num_unique_nodes}, num_edges={full_data.num_interactions}")

    for name, d in datasets.items():
        unique_nodes = np.unique(np.hstack([d.src_node_ids, d.dst_node_ids])).shape[0]
        print(f"{name}: unique_nodes={unique_nodes}, num_edges={d.src_node_ids.shape[0]}")
    print("\n")

    for name, d in [("train", train_data), ("val", val_data), ("new_node_val", new_node_val_data)]:
        node_ratio = d.num_unique_nodes / full_data.num_unique_nodes if full_data.num_unique_nodes > 0 else float('nan')
        edge_ratio = d.num_interactions / full_data.num_interactions if full_data.num_interactions > 0 else float('nan')
        print(f"{name}: node_ratio={node_ratio:.4f}, edge_ratio={edge_ratio:.4f}")
    print("\n")

    for name, pyg_d in pyg_datasets.items():
        has_nan = torch.isnan(pyg_d.x).any().item()
        print(f"pyg_{name}: x contains nan? {has_nan}")
    
    print("-" * 50, "\n\n")