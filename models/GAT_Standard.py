from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT, GraphSAGE, GCN
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from torch import nn
import torch
import numpy as np
from models.modules import TimeEncoder
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import add_self_loops


class GAT_Simple(nn.Module): 
    def __init__(self, heads, num_layers, negative_slope, add_self_loops, time_dim, device, **kwargs):
        super(GAT_Simple, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = kwargs['in_channels'] if i == 0 else kwargs['out_channels'] * heads
            out_channels = kwargs['out_channels'] if i== 0 else kwargs['out_channels']//heads
            dropout = kwargs.get('dropout', 0.0)
            self.layers.append(
                GATConv(in_channels, out_channels, heads=heads, negative_slope=negative_slope,
                        add_self_loops=add_self_loops, dropout=dropout)
            )
        self.device = device
        self.time_encoder = TimeEncoder(time_dim)
    
    def compute_src_dst_node_temporal_embeddings(self, batch_data: PyGData, edge_weight=None):
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
        edge_ids = batch_data.edge_attr
        edge_weight = batch_data.pos

        
        # Encode time and optionally concatenate edge weights
        edge_feature = torch.abs(self.time_encoder(timestamps).to(self.device).squeeze(1))
        edge_index, edge_feature = add_self_loops(edge_index=edge_index, edge_attr=edge_feature, fill_value=1.0, num_nodes=x.size(0))
        
        # Pass through GAT layers
        for layer in self.layers:
            x = layer.forward(x, edge_index, edge_attr=edge_feature)
        return x

    def forward_predict(self, node_feat, edge_index):
        
        for layer in self.layers:
            x = layer.forward(node_feat, edge_index)
            
        return x
    

if __name__ == "__main__":
    # Example usage
    x = torch.randn(10, 16)  # 10 nodes with 16 features each
    src_node_ids = np.array([0, 1, 2, 3])
    dst_node_ids = np.array([4, 5, 6, 7])
    timestamps = np.array([1.0, 2.0, 3.0, 4.0])
    
    time_dim = 16  # Example time encoder
    kwargs = {
        'in_channels': 16,
        'out_channels': 12,
        'dropout': 0.3,
        'num_layers': 2,
    }
    model = GAT_Simple(heads=2, negative_slope=0.2, add_self_loops=True, time_dim=time_dim, **kwargs)
    
    src_emb, dst_emb = model.compute_src_dst_node_temporal_embeddings(x, src_node_ids, dst_node_ids, timestamps)
    print("Source Node Embeddings:", src_emb.shape, src_emb.device, src_emb.dtype)
    print("Destination Node Embeddings:", dst_emb.shape, dst_emb.device, dst_emb.dtype)