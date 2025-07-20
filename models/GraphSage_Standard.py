from torch_geometric.nn import SAGEConv
from torch import nn
import torch
import numpy as np
from models.modules import TimeEncoder
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import add_self_loops

class GraphSage_Simple(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, dropout, device, time_dim: int, **kwargs):
        super(GraphSage_Simple, self).__init__(**kwargs)
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, out_channels, normalize=True)
        self.time_encoder = TimeEncoder(time_dim)
        self.device = device
        self.drop_rate = dropout
    
    def compute_src_dst_node_temporal_embeddings(self, batch_data, edge_weight=None):
        """
        Compute the temporal embeddings for source and destination nodes.
        :param x: Node features of Entire Graph.
        :param src_node_ids: Source node indices (numpy array).
        :param dst_node_ids: Destination node indices (numpy array).
        :param timestamps: Edge timestamps (numpy array).
        :param edge_weight: Edge weights (optional, numpy array or tensor).
        :return: Tuple of source and destination node embeddings.
        """
        # Create unique node idx list and mapping
        x = batch_data.x
        edge_index = batch_data.edge_index
        timestamps = batch_data.time
        
        # Encode time and optionally concatenate edge weights
        edge_feature = torch.abs(self.time_encoder(timestamps).to(self.device).squeeze(1))
        edge_index, edge_feature = add_self_loops(edge_index=edge_index, edge_attr=edge_feature, fill_value=1.0, num_nodes=x.size(0))
        
        # Pass through GraphSage layers
        x1 = self.sage1.forward(x, edge_index)
        x1_dot = torch.sigmoid(x1)
        x1_drop = torch.nn.functional.dropout(x1_dot, p=self.drop_rate, training=self.training)
        
        x2 = self.sage2.forward(x1_drop, edge_index)
        x2_dot = torch.sigmoid(x2)
        x2_drop = torch.nn.functional.dropout(x2_dot, p=self.drop_rate, training=self.training)

        return x2_drop
    
    def forward_predict(self, node_feat, edge_index):
        """
        Forward pass for link prediction.
        :param node_feat: Node features.
        :param edge_index: Edge indices.
        :return: Node embeddings after GraphSage layers.
        """
        x1 = self.sage1.forward(node_feat, edge_index)
        x1_dot = torch.sigmoid(x1)
        x1_drop = torch.nn.functional.dropout(x1_dot, p=self.drop_rate, training=self.training)

        x2 = self.sage2.forward(x1_drop, edge_index)
        x2_dot = torch.sigmoid(x2)
        x2_drop = torch.nn.functional.dropout(x2_dot, p=self.drop_rate, training=self.training)
        return x2_drop