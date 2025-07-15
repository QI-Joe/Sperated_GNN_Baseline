import pandas as pd
import numpy as np
import torch
import random
import math
from torch_geometric.data import Data
from torch import Tensor
import copy
import os
from torch_geometric.loader import NeighborLoader
from typing import Any, Union, Tuple
from multipledispatch import dispatch
import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
from datetime import datetime
import itertools
from collections import defaultdict
import enum
from sklearn.preprocessing import scale

MOOC, Mooc_extra = "Temporal_Dataset/act-mooc/act-mooc/", ["mooc_action_features", "mooc_action_labels", "mooc_actions"]
MATHOVERFLOW, MathOverflow_extra = "Temporal_Dataset/mathoverflow/", ["sx-mathoverflow-a2q", "sx-mathoverflow-c2a", "sx-mathoverflow-c2q", "sx-mathoverflow"]
OVERFLOW = r"../Standard_Dataset/lp/"
STATIC = ["mathoverflow", "askubuntu", "stackoverflow", "mooc"]
DYNAMIC = ["mathoverflow", "askubuntu", "stackoverflow"]
KG = ["entities"]

class NodeIdxMatching(object):

    """
    Not that appliable on train_mask or node_mask, at least in CLDG train_mask should be aggreated within NeigborLoader
    as seed node, while it is computed manually to present "positive pair"
    """

    def __init__(self, is_df: bool, df_node: pd.DataFrame = None, nodes: np.ndarray = [], label: np.ndarray=[]) -> None:
        """
        self.node: param; pd.Dataframe has 2 columns,\n
        'node': means node index in orginal entire graph\n
        'label': corresponding label
        """
        super(NodeIdxMatching, self).__init__()
        self.is_df = is_df
        if is_df: 
            if not df_node: raise ValueError("df_node is required")
            self.node = df_node
        else:
            if not isinstance(nodes, (np.ndarray, list, torch.Tensor)): 
                nodes = list(nodes)
            if len(label) > len(nodes):
                label = np.arange(len(nodes))
            self.nodes = self.to_numpy(nodes)
            self.node: pd.DataFrame = pd.DataFrame({"node": nodes, "label": label}).reset_index()

    def to_numpy(self, nodes: Union[torch.Tensor, np.array]):
        if isinstance(nodes, torch.Tensor):
            if nodes.device == "cuda:0":
                nodes = nodes.cpu().numpy()
            else: 
                nodes = nodes.numpy()
        return nodes

    def idx2node(self, indices: Union[np.array, torch.Tensor]) -> np.ndarray:
        indices = self.to_numpy(indices)
        node = self.node.node.iloc[indices]
        return node.values
    
    def node2idx(self, node_indices: Union[np.array, torch.Tensor] = None) -> np.ndarray:
        if node_indices is None:
            return np.array(self.node.index)
        node_indices = self.to_numpy(node_indices)
        indices = self.node.node[self.node.node.isin(node_indices)].index
        return indices.values
    
    def edge_restore(self, edges: torch.Tensor, to_tensor: bool = False) -> Union[torch.Tensor, np.array]:
        edges = self.to_numpy(edges)
        df_edges = pd.Series(edges.T).apply(lambda x: x.map(self.node.node))
        if to_tensor: 
            df_edges = torch.tensor(df_edges.values.T)
        return df_edges.values
    
    def edge_replacement(self, df_edge: pd.DataFrame):
        if not isinstance(df_edge, pd.Series):
            df_edge = self.to_numpy(df_edge)
            df_edge = pd.DataFrame(df_edge.T)
        transfor_platform = self.node.node
        transfor_platform = pd.Series(transfor_platform.index, index= transfor_platform.values)
        # given function "map" and data series, iterate through col is the fastest way
        df_edge = df_edge.apply(lambda x: x.map(transfor_platform))
        return df_edge.values.T
    
    def get_label_by_node(self, node_indices: Union[torch.Tensor, list[int], np.ndarray]) -> list:
        node_indices = self.to_numpy(node_indices)
        idx_mask: pd.Series = self.node.node[self.node.node.isin(node_indices)].index
        labels: list = self.node.label[idx_mask].values.tolist()
        return labels
    
    def get_label_by_idx(self, idx: Union[torch.Tensor, list[int], np.ndarray]) -> list:
        idx = self.to_numpy(idx)
        return self.node.label[idx].tolist()
    
    def sample_idx(self, node_indices: Union[torch.Tensor, list[int], np.ndarray]) -> torch.Tensor:
        node_indices = self.to_numpy(node_indices)
        idx_mask: pd.Series = self.node.node[self.node.node.isin(node_indices)].index
        return list(idx_mask.values)
    
    def matrix_edge_replacement(self, src: Union[pd.DataFrame|torch.Tensor])->np.ndarray:
        nodes = self.node["node"].values
        match_list = self.node[["node", "index"]].values

        max_size = max(nodes) + 1
        space_array = np.zeros((max_size,), dtype=np.int32)
        idx = match_list[:, 0]
        values = match_list[:, 1]

        space_array[idx] = values


        given_input = copy.deepcopy(src.T)

        col1, col2 = given_input[:, 0], given_input[:, 1]
        replace_col1 = space_array[col1]
        replced_col2 = space_array[col2]
        replaced_given = np.vstack((replace_col1, replced_col2)) # [2, n]
        return replaced_given


class Temporal_Dataloader(Data):
    """
    an overrided class of Data, to store splitted temporal data and reset their index 
    which support for fast local idx and global idx mapping/fetching
    """
    def __init__(self, nodes: Union[list, np.ndarray], edge_index: np.ndarray,\
                  edge_attr: Union[list|np.ndarray], y: list, edge_ids: np.ndarray, \
                    pos: tuple[torch.Tensor]) -> None:
        
        super(Temporal_Dataloader, self).__init__(x = pos[0], edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
        self.nodes = nodes
        self.edge_index = edge_index
        self.ori_edge_index = edge_index
        self.edge_ids = edge_ids
        self.edge_attr = edge_attr
        self.y = y
        self.kept_train_mask = None
        self.kept_val_mask = None

        self.node_pos, self.edge_pos = pos
        self.my_n_id = NodeIdxMatching(False, nodes=self.nodes, label=self.y)
        self.idx2node = self.my_n_id.node
        self.layer2_n_id: pd.DataFrame = None

    def side_initial(self):
        self.nn_val_mask, self.nn_test_mask = None, None
    
    def train_val_mask_injection(self, train_mask: np.ndarray, val_mask: np.ndarray, nn_val_mask):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.nn_val_mask = nn_val_mask
    
    def test_mask_injection(self, nn_test_mask: np.ndarray):
        self.nn_test_mask = nn_test_mask


class Dynamic_Dataloader(object):
    """
    a class to store a group of temporal dataset, calling with updated event running
    return a temporal data every time
    """
    def __init__(self, data: list[Data], graph: Data) -> None:
        super(Dynamic_Dataloader, self).__init__()
        self.data = data
        self.graph = graph
        self.num_classes = int(self.graph.y.max().item() + 1)
        self.len = len(data)

        self.num_nodes = self.graph.x.shape[0]
        self.num_edges = self.graph.edge_index.shape[-1]
        self.temporal = len(data)

        self.temporal_event = None

    def __getitem__(self, idx)-> Temporal_Dataloader:
        return self.data[idx]
    
    def get_temporal(self) -> Union[Data|Temporal_Dataloader|None]:
        if not self.temporal_event:
            self.update_event()
        return self.temporal_event
    
    def get_T1graph(self, timestamp: int) -> Union[Data|Temporal_Dataloader|None]:
        if timestamp>=self.len-1:
            return self.data[self.len-1]
        if len(self.data) <= 1:
            if self.data.is_empty(): return self.graph
            return self.data[0]
        return self.data[timestamp+1]

    def update_event(self, timestamp: int = -1):
        if timestamp>=self.len-1:
            return
        self.temporal_event = self.data[timestamp+1]


class Temporal_Splitting(object):

    class Label(enum.Enum):
        c1 = 1
        c2 = 2
        c3 = 3

    def __init__(self, graph: Data) -> None:
        
        super(Temporal_Splitting, self).__init__()
        self.graph = graph 

        if self.graph.edge_attr.any() == None:
            self.graph.edge_attr = np.arange(self.graph.edge_index.shape[0])

        self.n_id = NodeIdxMatching(False, nodes=self.graph.x, label=self.graph.y)
        self.temporal_list: list[Temporal_Dataloader] = []
        self.set_mapping: dict = None
    
    @dispatch(int, bool)
    def __getitem__(self, idx: int, is_node:bool = False):
        if is_node:
            return self.tracing_dict[idx]
        return self.temporal_list[idx]
    
    @dispatch(int, int)
    def __getitem__(self, list_idx: int, idx: int):
        return self.temporal_list[list_idx][idx]

    def sampling_layer(self, snapshots: int, views: int, span: float, strategy: str="sequential"):
        T = []
        if strategy == 'random':
            T = [random.uniform(0, span * (snapshots - 1) / snapshots) for _ in range(views)]
        elif strategy == 'low_overlap':
            if (0.75 * views + 0.25) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.75 * views + 0.25) * span /  snapshots)
            T = [start + (0.75 * i * span) / snapshots for i in range(views)]
        elif strategy == 'high_overlap':
            if (0.25 * views + 0.75) > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
            start = random.uniform(0, span - (0.25 * views + 0.75) * span /  snapshots)
            T = [start + (0.25 * i * span) / snapshots for i in range(views)]
        elif strategy == "sequential":
            T = [span * i / (snapshots-1) for i in range(1, snapshots)]
            if views > snapshots:
                return "The number of sampled views exceeds the maximum value of the current policy."
        
        T = random.sample(T, views)
        T= sorted(T)
        if T[0] == float(0):
            T.pop(0)
        return T

    def sampling_layer_by_time(self, span, duration: int = 30):
        """
        span :param; entire timestamp, expected in Unix timestamp such as 1254192988
        duration: param; how many days
        """
        Times = [datetime.fromtimestamp(stamp).strftime("%Y-%m-%d") for stamp in span]
        start_time = Times[0]

        T_duration: list[int] = []

        for idx, tim in enumerate(span):
            if Times[idx] - start_time >=duration:
                T_duration.append(tim)
                start_time = Times[idx]
        
        return T_duration

    def temporal_splitting(self, time_mode: str, **kwargs) -> list[Data]:
        """
        currently only suitable for CLDG dataset, to present flexibilty of function Data\n
        log 12.3:\n
        Temporal and Any Dynamic data loader will no longer compatable with static graph
        here we assume edge_attr means time_series in default
        """
        edge_index = self.graph.edge_index
        edge_attr = self.graph.edge_attr
        # pos = self.graph.pos
        edge_ids = np.arange(edge_index.shape[1], dtype=np.int64)

        max_time = max(edge_attr)
        temporal_subgraphs = []

        T: list = []

        if time_mode == "time":
            span = edge_attr.cpu().numpy()
            duration = kwargs["duration"]
            T = self.sampling_layer_by_time(span = span, duration=duration)
        elif time_mode == "view":
            span = (max(edge_attr) - min(edge_attr)).item()
            snapshot, views = kwargs["snapshot"], kwargs["views"]
            T = self.sampling_layer(snapshot, views, span)

        for idx, start in enumerate(T):
            if start<0.01: continue

            sample_time = start

            end = min(start + span / snapshot, max_time)
            sample_time = (edge_attr <= end) # returns an bool value

            sampled_edges = edge_index[:, sample_time]
            sampled_nodes = np.unique(sampled_edges) # orignal/gobal node index

            y = self.n_id.get_label_by_node(sampled_nodes)
            y = np.array(y)
            sub_edge_ids = edge_ids[sample_time]

            temporal_subgraph = Temporal_Dataloader(nodes=sampled_nodes, edge_index=sampled_edges, \
                edge_attr=edge_attr[sample_time], y=y, pos=(None, None), edge_ids=sub_edge_ids) # .get_Temporalgraph()
            
            temporal_subgraphs.append(temporal_subgraph)

        return temporal_subgraphs


def time_encoding(timestamp: torch.Tensor, emb_size: int = 64):
    
    timestamps = torch.tensor(timestamp, dtype=torch.float32).unsqueeze(1)
    max_time = timestamps.max() if timestamps.numel() > 0 else 1.0  # Avoid division by zero
    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
    
    te = torch.zeros(len(timestamps), emb_size)
    te[:, 0::2] = torch.sin(timestamps / max_time * div_term)
    te[:, 1::2] = torch.cos(timestamps / max_time * div_term)
    
    return te

def position_encoding(max_len, emb_size)->torch.Tensor:
    pe = torch.zeros(max_len, emb_size)
    position = torch.arange(0, max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def load_dblp_interact(path: str = None, dataset: str = "dblp", *wargs) -> pd.DataFrame:
    edges = pd.read_csv(os.path.join("/mnt/d/CodingArea/Python/Depreciated_data/CLDG/Data/CLDG-datasets/", dataset, '{}.txt'.format(dataset)), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join('/mnt/d/CodingArea/Python/Depreciated_data/CLDG/Data/CLDG-datasets/', dataset, 'node2label.txt'), sep=' ', names=['node', 'label'])

    return edges, label

def load_mathoverflow_interact(path: str = MATHOVERFLOW, *wargs) -> pd.DataFrame:
    edges = pd.read_csv(os.path.join(path, "sx-mathoverflow"+".txt"), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join(path, "node2label"+".txt"), sep=' ', names=['node', 'label'])
    return edges, label

def get_combination(labels: list[int]) -> dict:
    """
    :param labels: list of unique labels, for overflow it is fixed as [1,2,3]
    :return: a dictionary that stores all possible combination of labels, usually is 6
    """
    unqiue_node = len(labels)

    combination: dict = {}
    outer_ptr = 0
    for i in range(1, unqiue_node+1):
        pairs = itertools.combinations(labels, i)
        for pair in pairs:
            combination[pair] = outer_ptr
            outer_ptr += 1
    return combination

def load_static_overflow(prefix: str, path: str=None, *wargs) -> tuple[Data, NodeIdxMatching]:
    dataset = "sx-"+prefix
    path = OVERFLOW + prefix + r"/static"
    edges = pd.read_csv(os.path.join(path, dataset+".txt"), sep=' ', names=['src', 'dst', 'time'])
    label = pd.read_csv(os.path.join(path, "node2label.txt"), sep=' ', names=['node', 'label'])
    return edges, label

def load_mooc(path:str=None) -> Tuple[pd.DataFrame]:
    feat = pd.read_csv(os.path.join(path, "mooc_action_features.tsv"), sep = '\t')
    general = pd.read_csv(os.path.join(path, "mooc_actions.tsv"), sep = '\t')
    edge_label = pd.read_csv(os.path.join(path, "mooc_action_labels.tsv"), sep = '\t')
    return general, feat, edge_label

def edge_load_mooc(dataset:str):
    auto_path = r"../Standard_Dataset/lp/act-mooc/act-mooc"
    edge, feat, label = load_mooc(auto_path)
    # for edge, its column idx is listed as ["ACTIONID", "USERID", "TARGETID", "TIMESTAMP"]
    edge = edge.values
    edge_idx, src2dst, timestamp = edge[:, 0], edge[:, 1:3].T, edge[:, 3]
    
    print(src2dst.dtype, src2dst.shape)
    src2dst = src2dst.astype(np.int64)
    
    edge_pos = feat.iloc[:, 1:].values
    y = label.iloc[:, 1].values
    
    node = np.unique(src2dst).astype(np.int64)
    max_node = int(np.max(node)) + 1
    if dataset == "mooc":
        node = np.unique(src2dst[0])
    node_pos = position_encoding(max_node, 64).numpy()
    # edge_pos = time_encoding(timestamp)
    
    pos = (node_pos, edge_pos)
    graph = Data(x = node, edge_index=src2dst, edge_attr=timestamp, y = y, pos = pos)
    return graph

def load_dynamic_overflow(prefix: str, path: str=None, *wargs) -> tuple[pd.DataFrame, dict]:
    dataset = prefix
    path = OVERFLOW + prefix + r"/dynamic"
    labels: list = [1,2,3]
    edges = pd.read_csv(os.path.join(path, dataset+".txt"), sep=' ', names=['src', 'dst', 'time', 'appearance'])
    combination_dict = get_combination(labels)
    
    return edges, combination_dict

def dynamic_label(edges: pd.DataFrame, combination_dict: dict) -> pd.DataFrame:
    """
    Very slow when facing large dataset. Recommend to use function matrix_dynamic_label
    """
    unique_node = edges[["src", "dst"]].stack().unique()
    node_label: list[tuple[int, int]] = []
    for node in unique_node:
        appearance = edges[(edges.src == node) | (edges.dst == node)].apprarance.values
        appearance = tuple(set(appearance))
        node_label.append((node, combination_dict[appearance]))
    return pd.DataFrame(node_label, columns=["node", "label"])


def load_KG_dataset(load_: dict) -> tuple[Temporal_Dataloader, NodeIdxMatching]:
    """
    Load KG dataset, node feature is embedded data from Bert
    """
    node_path, edge_path = os.path.join(load_["general"], load_["node"]), os.path.join(load_["general"], load_["edge"])
    node_csv, edge_csv = pd.read_csv(node_path), pd.read_csv(edge_path)
    node_csv.columns = ["idx", "semantuic", "label"]
    edge_csv.columns = ["src", "dst", "type", "relation"]
    
    node_feat, edge_feat = os.path.join(load_["general"], load_["node_feat"]), os.path.join(load_["general"], load_["edge_feat"])
    np_node, np_edge = np.load(node_feat, allow_pickle=True), np.load(edge_feat, allow_pickle=True)
    
    nodes = node_csv["idx"].values
    edge_index = edge_csv[["src", "dst"]].values.T
    timestamp = edge_csv.index.values
    label = node_csv["label"].values
    pos = (node_feat, edge_feat)
    
    graph = Data(x=nodes, edge_index=edge_index, edge_attr=timestamp, y=label, pos=pos)
    return graph, None
    

def load_static_dataset(path: str = None, dataset: str = "mathoverflow", emb_size: int = 64, **wargs) -> tuple[Temporal_Dataloader, NodeIdxMatching]:
    """
    Now this txt file only limited to loading data in from mathoverflow datasets
    path: (path, last three words of dataset) -> (str, str) e.g. ('data/mathoverflow/sx-mathoverflow-a2q.txt', 'a2q')
    node Idx of mathoverflow is not consistent!
    """
    fea_dim = emb_size
    if dataset[-8:] == "overflow" or dataset == "askubuntu":
        edges, label = load_static_overflow(dataset) if not path else load_static_overflow(dataset, path)
    elif dataset == "dblp":
        edges, label = load_dblp_interact() if not path else load_dblp_interact(path)
    elif dataset == "mooc":
        return edge_load_mooc(dataset), None

    x = label.node.to_numpy()
    nodes = position_encoding(x.max()+1, fea_dim)[x].numpy()
    labels = label.label.to_numpy()

    edge_index = edges.loc[:, ["src", "dst"]].values.T
    start_time = edges.time.min()
    edges.time = edges.time.apply(lambda x: x - start_time)
    time = edges.time.values
    
    time_pos = time_encoding(timestamp=time).numpy()
    pos = (nodes, time_pos)

    graph = Data(x=x, edge_index=edge_index, edge_attr=time, y=labels, pos = pos)
    # neighborloader = NeighborLoader(graph, num_neighbors=[10, 10], batch_size =2048, shuffle = False)
    idxloader = NodeIdxMatching(False, nodes=x, label=labels)
    return graph, idxloader

def load_tsv(path: list[tuple[str]], *wargs) -> tuple[pd.DataFrame]:
    """
    Note this function only for loading data in act-mooc dataset
    """
    dfs:dict = {p[1]: pd.read_csv(p[0], sep='\t') for p in path}

    label = dfs["mooc_action_labels"]
    action_features = dfs["mooc_action_features"]
    actions = dfs["mooc_actions"]
    return label, action_features, actions

def load_example():
    return "node_feat", "node_label", "edge_index", "train_indices", "val_indices", "test_indices"

def data_load(dataset: str, **wargs) -> tuple[Temporal_Dataloader, Union[NodeIdxMatching|dict]]:
    dataset = dataset.lower()
    if dataset in STATIC:
        return load_static_dataset(dataset=dataset, **wargs)
    elif dataset in KG:
        path = wargs.get("path", None)
        return load_KG_dataset(path)
    raise ValueError("Dataset not found")

def t2t1_node_alignment(t_nodes: set, t: Temporal_Dataloader, t1: Temporal_Dataloader) -> Tuple[np.ndarray, np.ndarray, int]:
    t_list = t.my_n_id.node.values      # index (resort node index), global node idx, label
    t1_list = t1.my_n_id.node.values    # index (resort node index), global node idx, label

    t2t1 = t_list[np.isin(t_list[:, 0], list(t_nodes)), 1].tolist()
    t1_extra = list(set(t1_list[:,1]) - set(t_list[:,1]))

    new_nodes = sorted(set(t2t1+t1_extra)) # be set or not be doesnt matter, duplication wont lead to different mask
    resort_nodes = t1_list[np.isin(t1_list[:,1], new_nodes), 0].tolist()

    t1_src = np.isin(t1.edge_index[0], resort_nodes)
    t1_dst = np.isin(t1.edge_index[1], resort_nodes)

    return t1_src*t1_dst, ~t1_src*~t1_dst, len(new_nodes)