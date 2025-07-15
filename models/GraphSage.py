import copy
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random
from torch_geometric.data import Data
from collections import defaultdict
from torch_geometric.loader import NeighborLoader
from typing import Union


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model: Union[nn.Module|None]=None, gcn=False, to_cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator: MeanAggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.to_cuda = to_cuda
        # self.aggregator.to_cuda = to_cuda
        if to_cuda:
            self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim).cuda(), requires_grad=True)
        else:
            self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim), requires_grad=True)
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)
        if not self.gcn:
            if self.to_cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined


def flatten(l):
    return [item for sublist in l for item in sublist]

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, to_cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.to_cuda = to_cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        global device
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(sorted(to_neigh), 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.to_cuda:
            mask = mask.to(device)
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.to_cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).to(device))
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

    

class SupervisedGraphSage(nn.Module):

    def __init__(self):
        super(SupervisedGraphSage, self).__init__()
        self.enc: Encoder = None
        self.weight = None
        self.xent = nn.CrossEntropyLoss()

        self.last_base_model_features = None
        self.last_base_model_features_aggregator = None
        self.past_adj_lists = None

    def enc2_load(self, num_classes, enc: Encoder):
        self.enc = enc
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim)).cuda()
        init.xavier_uniform(self.weight)
        # self.weight = self.weight.cuda()

    def cached_exam(self):
        if self.last_base_model_features is None or self.past_adj_lists is None or self.last_base_model_features_aggregator is None:
            return False
        return True
    
    def load_cache(self):
        return self.past_adj_lists, self.last_base_model_features
    
    def reset_parameters(self):
        init.xavier_uniform(self.weight)

    def t_moment_redress(self):
        r"""
        Used for update, store, and switch the embedding of aggregator 1 and encoder 1
        """
        tmp_adj = self.enc.adj_lists
        tmp_features_enc = self.enc.base_model.features
        tmp_features_agg = self.enc.base_model.aggregator.features

        self.enc.base_model.features = self.last_base_model_features
        self.enc.base_model.aggregator.features = self.last_base_model_features_aggregator
        self.enc.adj_lists = self.past_adj_lists
        self.enc.base_model.adj_lists = self.past_adj_lists

        self.last_base_model_features = tmp_features_enc
        self.last_base_model_features_aggregator = tmp_features_agg
        self.past_adj_lists = tmp_adj

    def encoder_adj_redress(self, new_embedding: torch.Tensor, adj_lists: dict):
        r"""
        Used for update, store, and switch the embedding of aggregator 1 and encoder 1

        :param new_embedding is a detached new temporal given feature of embeddings, it expected that in shape (N, D)
        new_embedding requries 
        """
        num_nodes, embed_dim = new_embedding.shape
        new_embedding = new_embedding.detach().cpu()
        Pcontainer = nn.Embedding(num_nodes, embed_dim)
        Pcontainer.weight = nn.Parameter(torch.FloatTensor(new_embedding), requires_grad=False)
        Pcontainer = Pcontainer.to("cuda:0")

        self.last_base_model_features = copy.deepcopy(self.enc.base_model.features)
        self.last_base_model_features_aggregator = copy.deepcopy(self.enc.base_model.aggregator.features)

        self.past_adj_lists = self.enc.adj_lists

        # base model, suppose to be ptr instead of a new deep copy object
        self.enc.base_model.features = Pcontainer
        self.enc.base_model.aggregator.features = Pcontainer

        self.enc.base_model.adj_lists = adj_lists
        self.enc.adj_lists = adj_lists

    def forward(self, nodes):
        embeds = self.enc.forward(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        """
        from model strcuture it seems use Neighrborloader is a bit redundant here,
        """
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
    
def adjacent_list_building(graph: Data)->dict:
    r"""
    exclusively served for GraphSage neighbor node connection, plan to extent for node size convertion
    :param graph: Temporal_Dataloader object
    :return: a dictionary converted and able to match with temporal and entire graph
    """
    if isinstance(graph.edge_index, torch.Tensor):
        edges = graph.edge_index.cpu().numpy().T
    else:
        edges = graph.edge_index.T
    adj_lists = defaultdict(set)
    for idx, edge in enumerate(edges):
        src, dst = edge
        adj_lists[src].add(dst)
        adj_lists[dst].add(src)
    return adj_lists

def node_splitation(data: Data, train_ratio: float = 0.1, val_ratio:float = 0.1) -> tuple:
    nodes = data.x
    num_nodes = len(nodes)
    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio should be less than 1")
    train_num = int(num_nodes * train_ratio)
    val_num = int(num_nodes * (val_ratio + train_ratio))
    train_node = nodes[:train_num]
    val_node = nodes[train_num: val_num]
    if train_ratio + val_ratio == 1:
        return train_node, val_node, None
    test_node = nodes[val_num:]
    return train_node, val_node, test_node