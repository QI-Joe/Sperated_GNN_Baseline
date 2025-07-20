import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.GCA_functional import SimpleParam
import argparse
import torch

def data_clean(target: list[object]):
    for obj in target:
        del obj
    return 

class Node_Affinity_Gen(object):
    """
    It is expected that all income node/graph should be list in a certain pd.DataFrame format \n
    OK, the class wont hold functionality of customized data splitation and weight computing, \n
    but more focus on build-up a simple work
    """
    def __init__(self, node_frame: pd.DataFrame, edge_frame: pd.DataFrame):
        """
        node_frame: a pd.Dataframe owns columns of original node_idx in entire graph and resorted node_idx \n
        edge_frame: a pd.Dataframe owns column of edge_idx(rematched) E, edge_weight W, timestamp T, with order of \n
        edge_idx u: column 0, edge_idx i: column 1, edge_weight: column 2, (optional)timestamp: column 3
        """
        super(self, Node_Affinity_Gen).__init__()

        self.nodes: pd.DataFrame = node_frame
        self.edges: pd.DataFrame = edge_frame

        columns = edge_frame.columns
        self.eu: int = columns[0]
        self.ei: int = columns[1]
        self.ew: int = columns[2]

    def node_reduce(self, numerator: pd.DataFrame, denominator: pd.DataFrame) -> tuple[defaultdict]:
        dict_num: defaultdict = defaultdict(float, {(u, i): w for u,i,w in numerator.values})
        dict_denom: defaultdict = defaultdict(float, {u:w for u, w in denominator.values})
        return dict_num, dict_denom

    def node_map(self, key_value: pd.DataFrame) -> tuple[defaultdict]:
        numerator = key_value.groupby([self.eu, self.ei], as_index=False)[self.ew].sum()
        denominator = key_value.groupby([self.eu], as_index=False)[self.ew].sum()

        return self.node_reduce(numerator, denominator)

    def node_label_gen(self)->pd.DataFrame:
        self.numerator, self.denominator = self.node_map(self.edges)
        storage_ = list()
        for (u, i), w in self.numerator.items():
            u_denom = self.denominator[u]
            storage_.append([u, i, w/u_denom])
        return pd.DataFrame(storage_, columns=["u", "i", "label"])

# have bugs
def ordered_sets(ptr_idx_list: list[int], input_sets):
    upper_bound = len(input_sets)
    if len(ptr_idx_list)==1:
        yield list(range(ptr_idx_list[0], upper_bound+1))
    for idx in range(ptr_idx_list[0], upper_bound-len(ptr_idx_list[1:])):
        ptr_idx_list[1] = idx+1
        last_sample = ordered_sets(ptr_idx_list[1:], input_sets)
        last_sample = next(last_sample)
        prefix = [ptr_idx_list[0]]*len(last_sample)
        yield [combine + last_ for combine, last_ in zip(prefix, last_sample)]

def sets_operation(input_sets: list[set[int]]):
    sets_order_dict: dict[tuple[int]: int] = dict()
    for ptr in range(1, len(input_sets)):
        ptr_idx_list = list(range(ptr))
        sets_order_dict = ordered_sets(ptr_idx_list, input_sets)
    return sets_order_dict

def get_param_dict(dataset_name: str, param: str = "local:coauthor_cs.json"):
    default_param = {
        'learning_rate': 0.05,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 1500,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }
    sp = SimpleParam(default=default_param)
    param = sp(source=param, preprocess='nni')
    return param

def get_link_prediction_args():
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='entities',
                        choices=['dblp', 'mooc', 'entities', 'askubuntu'])
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--model_name', type=str, default='GCN', help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=["MVGRL", "GCA", "GCN", "GAT", "GraphSAGE"])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--snapshot', type=int, default=8, help='how many snapshots of graphs there are')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--hidden_channels', type=int, default=64, help='number of hidden channels')
    parser.add_argument('--weight_lambda', type=float, default=0.8, help='weight lambda for the loss function')
    parser.add_argument('--base_model', type=str, default='GCNConv', choices=['GCNConv', 'SAGEConv', 'GATConv'],
                        help='base model for the encoder, only applicable for GCA')
    parser.add_argument('--activation_function', type=str, default='prelu', choices=['relu', 'prelu', 'leaky_relu'],
                        help='activation function for the encoder, only applicable for GCA')
    parser.add_argument('--tau', type=float, default=0.4, help='temperature parameter for the contrastive loss')
    parser.add_argument('--negative_slope', type=float, default=0.0, help='GAT used parameter')
    parser.add_argument("--dataset_start", type=int, default=0, help="Start index for dataset.")
    parser.add_argument("--dataset_end", type=int, default=400, help="End index for dataset.")

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args