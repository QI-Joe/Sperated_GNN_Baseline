from torch_geometric.nn import GCNConv
from torch.nn import init
import torch
import torch.nn.functional as F
from models.modules import TimeEncoder
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data as PyGData

class GCONV_Simple(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim1, hidden_dim2, device, time_dim: int, dropout=0.0):
        super(GCONV_Simple, self).__init__()
        # Architecture:
        # 2 MLP layers to preprocess BERT repr,
        # 2 GCN layer to aggregate node embeddings
        self.input_dim = in_channels
        self.output_dim = out_channels
        self.device = device
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.FullConnected_MLP()
        self.Praser_GCN()
        self.time_encoder = TimeEncoder(time_dim).to(self.device)

    
    def FullConnected_MLP(self):
        self.layer1 = torch.nn.Linear(self.input_dim, self.hidden_dim1).to(self.device)
        self.layer1.weight = torch.nn.Parameter(torch.FloatTensor(self.input_dim, self.hidden_dim1).t().to(self.device))
        init.xavier_uniform_(self.layer1.weight)

    def Praser_GCN(self):
        self.conv1 = GCNConv(self.hidden_dim1, self.hidden_dim2).to(self.device)
        self.conv2 = GCNConv(self.hidden_dim2, self.output_dim).to(self.device)

    def compute_src_dst_node_temporal_embeddings(self, batch_data, edge_weight=None):
        node_feature = batch_data.x
        edge_index = batch_data.edge_index
        timestamp = batch_data.time
        
        x = self.layer1.forward(node_feature)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_feature = torch.abs(self.time_encoder(timestamp).to(self.device).squeeze(1))
        edge_index, edge_feature = add_self_loops(edge_index=edge_index, edge_attr=edge_feature, fill_value=1.0, num_nodes=x.size(0))

        x = self.conv1.forward(x, edge_index, edge_weight=edge_feature)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2.forward(x, edge_index, edge_weight=edge_feature)
        return F.log_softmax(x, dim=1)

    def forward_predict(self, node_feat, edge_index):
        """
        Forward pass for link prediction.
        :param node_feat: Node features.
        :param edge_index: Edge indices.
        :return: Node embeddings after GCN layers.
        """
        x = self.layer1.forward(node_feat)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv1.forward(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2.forward(x, edge_index)
        return x