from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
from typing import Optional, Any
from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting
import torch 
from numpy import ndarray
import copy
from utils.robustness_injection import Edge_Distrub, Imbalance, Few_Shot_Learning

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, \
                hash_table: dict[int, int] = None, node_feat: Optional[ndarray|None] = None, edge_feat: Optional[ndarray|None] = None, true_seen_label_mask: Optional[ndarray|None] = None):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.hash_table = hash_table
        
        self.target_node: Optional[set|None] = None
        self.seen_nodes: np.ndarray = None
        self.true_seen_label_mask = true_seen_label_mask

    def set_up_seen_nodes(self, seen_nodes: np.ndarray, indcutive_seen_nodes: np.ndarray):
        """
        set up the seen nodes
        :param seen_nodes: np.ndarray, seen nodes
        :param indcutive_seen_nodes: np.ndarray, inductive seen nodes
        """
        self.seen_nodes = seen_nodes
        self.target_node = indcutive_seen_nodes

    def setup_robustness(self, match_tuple: tuple[np.ndarray, np.ndarray], inductive_match_tuple: Optional[tuple[np.ndarray, np.ndarray]] = None):
        self.robustness_match_tuple = match_tuple
        self.inductive_match_tuple = inductive_match_tuple

def to_TPPR_Data(graph: Temporal_Dataloader, task: str = "node") -> Data:
    nodes = graph.x
    edge_idx = np.arange(graph.edge_index.shape[1])
    timestamp = graph.edge_attr
    src, dest = graph.edge_index[0, :], graph.edge_index[1, :]
    labels = graph.y

    hash_dataframe = copy.deepcopy(graph.my_n_id.node.loc[:, ["index", "node"]].values.T)
    
    """
    :param hash_table, should be a matching list, now here it is refreshed idx : origin idx,
    """
    if task == "link":
        hash_table: dict[int, int] = {idx: node for idx, node in zip(*hash_dataframe)}
    elif task == "node":
        hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}

    # edge_feat, node_feat = graph.edge_pos, graph.node_pos
    TPPR_data = Data(src_node_ids= src, dst_node_ids=dest, node_interact_times=timestamp, edge_ids = edge_idx, \
                     labels=labels, hash_table=hash_table)

    return TPPR_data

def span_time_quantile(threshold: float, tsp: np.ndarray, dataset: str):
    val_time = np.quantile(tsp, threshold)
    
    if dataset in ["dblp", "tmall"]:
        spans, span_freq = np.unique(tsp, return_counts=True)
        if val_time == spans[-1]: val_time = spans[int(spans.shape[0]*threshold)]
    return val_time

def get_link_prediction_data(dataset_name: str, snapshot: int, val_ratio: float = 0.8, test_ratio: float=0.0, kg_path: Optional[dict[str, str]] = None) -> tuple[np.ndarray, np.ndarray, list[Data]]:
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    
    Attention ! For Node Classification task, we need to refresh the node index!
    """
    view = snapshot - 2

    graph, idx_list = data_load(dataset=dataset_name, emb_size = 64, path=kg_path)
    node_raw_features, edge_raw_features = graph.pos

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    if dataset_name != "entities":    
        assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
        assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
    
    graph.pos = node_raw_features, edge_raw_features

    # assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'
    
    if dataset_name != "askubuntu":
        graph_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", snapshot=snapshot, views = view)
    else:
        graph_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", snapshot=snapshot+2, views = view+2)
        graph_list = graph_list[-view:]
    
    assert len(graph_list) == view, f"the number of views should be {view}, but got {len(graph_list)}"
    if dataset_name == "entities":
        view = 2
    
    Data_list = list()
    for idx in range(view-1):
        temporal_graph = graph_list[idx]
        # get the timestamp of validate and test set
        full_data = to_TPPR_Data(temporal_graph)
        src_node_ids = full_data.src_node_ids.astype(np.longlong)
        dst_node_ids = full_data.dst_node_ids.astype(np.longlong)
        node_interact_times = full_data.node_interact_times.astype(np.float64)
        edge_ids = full_data.edge_ids.astype(np.longlong)
        labels = full_data.labels
        
        val_time = span_time_quantile(threshold=0.8, tsp=node_interact_times, dataset=dataset_name)

        # the setting of seed follows previous works
        random.seed(2025)

        # union to get node set
        node_set = set(src_node_ids) | set(dst_node_ids)
        num_total_unique_node_ids = len(node_set)

        # compute nodes which appear at test time
        t1_temporal: Temporal_Dataloader = graph_list[idx+1]
        t1_full_data: Data = to_TPPR_Data(t1_temporal)
        t1_node_set = set(t1_full_data.src_node_ids) | set(t1_full_data.dst_node_ids)
        # t1_num_unique_node_ids = len(t1_node_set)
        # t_hash_table, t1_hash_table = full_data.hash_table, t1_full_data.hash_table
        
        """
        Basically, test_node_set should be valid node set consdiering its temporal features, lets make a row:
        new val test set = set(random.sample(val_ndoe_set, int(0.1 * val_node_set))
        """
        val_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
        new_val_node_set = set(random.sample(sorted(val_node_set), int(0.05 * num_total_unique_node_ids)))
        
        new_val_source_mask = np.isin(src_node_ids, sorted(new_val_node_set))
        new_val_destination_mask = np.isin(dst_node_ids, sorted(new_val_node_set))
        
        observed_edge_mask = np.logical_and(~new_val_source_mask, ~new_val_destination_mask)
        # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
        train_mask = np.logical_and(node_interact_times <= val_time, observed_edge_mask)
        # train_mask = node_interact_times<=val_time

        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                        node_interact_times=node_interact_times[train_mask],
                        edge_ids=edge_ids[train_mask], labels=labels)

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
        # assert len(train_node_set & new_val_node_set) == 0
        # new nodes that are not in the training set
        new_node_set = node_set - train_node_set # key points 1

        val_mask = node_interact_times > val_time

        # new edges with new nodes in the val and test set (for inductive evaluation)
        edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                                for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)

        # validation and test data
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                        node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels)

        test_data = t1_full_data

        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                                node_interact_times=node_interact_times[new_node_val_mask],
                                edge_ids=edge_ids[new_node_val_mask], labels=labels)
        """
        try to not resort the new node set, see whether it works
        """
        # t_node_match, t1_node_match = np.vectorize(t1_hash_table.get), np.vectorize(t_hash_table.get)
        # t_node_original, t1_node_original = t_node_match(sorted(train_node_set)), t1_node_match(sorted(t1_node_match))
        
        # t1_new_node_set = sorted(set(t1_node_original) - set(t_node_original)) # key points 2
        # reverse_hash_table = {v: k for k, v in t1_hash_table.items()}
        # t1_new_node_set = np.vectorize(reverse_hash_table.get)(t1_new_node_set)
        
        t1_new_node_set = t1_node_set - train_node_set # key points 2.1
        t1_edge_contains_new_node_mask = np.array([(src_node_id in t1_new_node_set or dst_node_id in t1_new_node_set)
                                                    for src_node_id, dst_node_id in zip(t1_full_data.src_node_ids, t1_full_data.dst_node_ids)])
        new_node_test_mask = np.logical_and(t1_full_data.node_interact_times, t1_edge_contains_new_node_mask)  
        
        new_node_test_data = Data(src_node_ids=test_data.src_node_ids[new_node_test_mask], dst_node_ids=test_data.dst_node_ids[new_node_test_mask],
                                node_interact_times=test_data.node_interact_times[new_node_test_mask],
                                edge_ids=test_data.edge_ids[new_node_test_mask], labels=test_data.labels)
        
        Data_list.append([full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data])

        print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
        print("The training dataset has {} interactions, involving {} different nodes, with ratio of {:.4f}".format(
            train_data.num_interactions, train_data.num_unique_nodes, train_data.num_interactions/full_data.num_interactions))
        print("The validation dataset has {} interactions, involving {} different nodes, with ratio of {:.4f}".format(
            val_data.num_interactions, val_data.num_unique_nodes, val_data.num_interactions / full_data.num_interactions))
        print("The test dataset has {} interactions, involving {} different nodes".format(
            test_data.num_interactions, test_data.num_unique_nodes))
        print("The new node validation dataset has {} interactions, involving {} different nodes, with nn_val/validation ratio {:.4f}".format(
            new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes, new_node_val_data.num_interactions / full_data.num_interactions))
        print("The new node test dataset has {} interactions, involving {} different nodes, with nn_test/(test-full_data) interaction ratio {:.4f}".format(
            new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes, new_node_test_data.num_interactions / (test_data.num_interactions - full_data.num_interactions)))
        print("{} nodes were used for the inductive testing, i.e. are never seen during training\n\n".format(len(t1_new_node_set)))

    return node_raw_features, edge_raw_features, Data_list # full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def fast_Data_object_update(match_table: pd.DataFrame, nodes: np.ndarray, full_data: Data, task:str = None) -> Data:
  """
  Updates a Data object by filtering its edges to include only those between specified nodes.
  Args:
    match_table (pd.DataFrame): A DataFrame containing at least three columns: ["index", "node", "label"]. Used to map new nodes to original nodes.
    nodes (np.ndarray): An array of indices representing the new nodes to be included.
    full_data (Data): The original Data object containing sources, destinations, timestamps, edge indices, labels, and optional attributes.
  Returns:
    Data: A new Data object containing only the edges where both source and destination nodes are among the specified nodes. Other attributes (labels, hash_table, node_feat) are preserved from the original Data object.
  """
  nptable = match_table.values # ["index", "node", "label"]
  original_node = nptable[nodes, 1] # nodes consisted by new node, use second column to match original node
  nn_src, nn_dst = np.isin(full_data.src_node_ids, original_node), np.isin(full_data.dst_node_ids, original_node)
  nn_mask = nn_src & nn_dst
  if task == "fsl":
    nn_mask = nn_src | nn_dst
  return Data(full_data.src_node_ids[nn_mask], full_data.dst_node_ids[nn_mask], full_data.node_interact_times[nn_mask],\
              full_data.edge_ids[nn_mask], full_data.labels, hash_table=full_data.hash_table, node_feat=full_data.node_feat)
  

def get_node_classification_data(dataset_name: str, snapshot: int, task:str, ratio: float = 0.0, val_ratio: float=0.8):
    r"""
    this function is used to convert the node features to the correct format\n
    e.g. sample node dataset is in the format of [node_id, edge_idx, timestamp, features] with correspoding\n
    shape [(n, ), (m,2), (m,), (m,d)]. be cautious on transformation method\n
    
    2025.4.5 TPPR and data_load method will not support TGB-Series data anymore
    """
    wargs = {"rb_task": task, "ratio": ratio}
    graph, _ = data_load(dataset_name, **wargs)
    node_raw_features, edge_raw_features = graph.pos
    node_cls = np.unique(graph.y)
    
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
    
    graph_list = Temporal_Splitting(graph).temporal_splitting(snapshot=snapshot, time_mode="view", views = snapshot-2)

    TPPR_list: list[list[Data]] = []
    lenth = len(graph_list) - 1 # no training for the last graph, so -1
        
    for idx in range(lenth):
      # covert Temproal_graph object to Data object
      items: Temporal_Dataloader = graph_list[idx]
      if task == "edge_disturb":
        print(f"Edge Random Deletion ratio is {ratio}")
        transformer = Edge_Distrub(ratio=ratio)
        items.edge_index, items.edge_attr = transformer(items)
      temporal_node_num = items.x.shape[0]
      
      src_edge = items.edge_index[0, :]
      dst_edge = items.edge_index[1, :]
      all_nodes = items.my_n_id.node["index"].values
      flipped_nodes = items.my_n_id.node["node"].values
      items.y = np.array(items.y)

      t_labels = items.y
      full_data = to_TPPR_Data(items)
      
      train_node, train_node_origin, train_label = all_nodes[:int(temporal_node_num*0.8)], flipped_nodes[:int(temporal_node_num*0.8)], t_labels[:int(temporal_node_num*0.8)]
      val_node, val_node_origin, val_label = all_nodes[int(temporal_node_num*0.8):], flipped_nodes[int(temporal_node_num*0.8):], t_labels[int(temporal_node_num*0.8):]
      
      train_selected_src_edge, train_selected_dst_edge = np.isin(src_edge, train_node_origin), np.isin(dst_edge, train_node_origin)
      train_mask = train_selected_src_edge & train_selected_dst_edge
      train_feature_should_be_seen = train_selected_src_edge | train_selected_dst_edge
      
      val_selected_src_edge, val_selected_dst_edge = np.isin(src_edge, val_node_origin), np.isin(dst_edge, val_node_origin)
      val_mask = val_selected_src_edge | val_selected_dst_edge
      nn_val_mask = val_selected_src_edge & val_selected_dst_edge
      
      hash_dataframe = copy.deepcopy(items.my_n_id.node.loc[:, ["index", "node"]].values.T)
      hash_table: dict[int, int] = {node: idx for idx, node in zip(*hash_dataframe)}
      
      train_data = Data(full_data.src_node_ids[train_mask], full_data.dst_node_ids[train_mask], full_data.node_interact_times[train_mask],\
                        full_data.edge_ids[train_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
      train_data.setup_robustness((train_node, train_label))
        
      val_data = Data(full_data.src_node_ids[val_mask], full_data.dst_node_ids[val_mask], full_data.node_interact_times[val_mask],\
                        full_data.edge_ids[val_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
      val_data.setup_robustness((val_node, val_label))
      
      train_data_edge_learn = Data(full_data.src_node_ids[train_feature_should_be_seen], full_data.dst_node_ids[train_feature_should_be_seen], \
      full_data.node_interact_times[train_feature_should_be_seen], full_data.edge_ids[train_feature_should_be_seen], t_labels, hash_table=hash_table, node_feat=full_data.node_feat)
      
      if nn_val_mask.sum() == 0:
        nn_val_data = copy.deepcopy(val_data)
      else:
        nn_val_data = Data(full_data.src_node_ids[nn_val_mask], full_data.dst_node_ids[nn_val_mask], full_data.node_interact_times[nn_val_mask],\
                            full_data.edge_ids[nn_val_mask], t_labels, hash_table = hash_table, node_feat=full_data.node_feat)
        nn_val_node_original = np.array(sorted(set(full_data.src_node_ids[nn_val_mask]) | set(full_data.dst_node_ids[nn_val_mask])))
        nn_val_node = np.vectorize(nn_val_data.hash_table.get)(nn_val_node_original)
        nn_val_data.setup_robustness((nn_val_node, t_labels[nn_val_node]))
      
      if task in ["imbalance", "fsl"]:
        if task == "imbalance":
          train_ratio, val_ratio = 0.8, None
          transform = Imbalance(train_ratio=train_ratio, ratio=ratio, val_ratio=val_ratio)
          items = transform(items)
        elif task == "fsl":
          transform = Few_Shot_Learning(fsl_num=ratio)
          items = transform(items)
      
        node_label = items.my_n_id.node["label"].values
          
        train_label, train_node = node_label[items.train_mask], all_nodes[items.train_mask]
        val_label, val_node = node_label[items.val_mask], all_nodes[items.val_mask]
        nn_val_label, nn_val_node = node_label[items.nn_val_mask], all_nodes[items.nn_val_mask]
        
        train_data = fast_Data_object_update(items.my_n_id.node, train_node, full_data, task=task)
        val_data = fast_Data_object_update(items.my_n_id.node, val_node, full_data, task=task)
        nn_val_data = fast_Data_object_update(items.my_n_id.node, nn_val_node, full_data, task=task)
        
        train_data.setup_robustness((train_node, train_label)) 
        val_data.setup_robustness((val_node, val_label))
        nn_val_data.setup_robustness((nn_val_node, nn_val_label))
            
      test: Temporal_Dataloader = graph_list[idx+1]
      test_data = to_TPPR_Data(test)
      test_node, test_label = test.my_n_id.node["index"].values, test.my_n_id.node["label"].values
      test_data.setup_robustness((test_node := test.my_n_id.node["index"].values, test_label))
      
      # test_data.setup_robustness((test_node, test_label))
      nn_test_node_original = np.array(sorted(set(test.my_n_id.node["node"].values) - set(flipped_nodes)))
      nn_test_node = np.vectorize(test_data.hash_table.get)(nn_test_node_original)
      nn_test_src, nn_test_dst = np.isin(test_data.src_node_ids, nn_test_node_original), np.isin(test_data.dst_node_ids, nn_test_node_original)
      nn_test_mask = nn_test_src | nn_test_dst
      nn_test_data = Data(test_data.src_node_ids[nn_test_mask], test_data.dst_node_ids[nn_test_mask], test_data.node_interact_times[nn_test_mask],\
                          test_data.edge_ids[nn_test_mask], test_label, hash_table = test_data.hash_table, node_feat=test_data.node_feat)
      
      nn_test_label = test_label[nn_test_node]
      nn_test_data.setup_robustness((nn_test_node, nn_test_label))
      
      if task in ["imbalance", "fsl"]:
        test_transform: Temporal_Dataloader = transform.test_processing(test)
        nn_test_match_list = (test_node[test_transform.nn_test_mask], test_label[test_transform.nn_test_mask])
        nn_test_data = fast_Data_object_update(test.my_n_id.node, nn_test_match_list[0], test_data, task=task)
        nn_test_data.setup_robustness(nn_test_match_list)

      node_num = items.num_nodes
      node_edges = items.num_edges

      TPPR_list.append([full_data, train_data, val_data, test_data, train_data_edge_learn, nn_val_data, nn_test_data, node_num, node_edges])
      
    return node_cls, node_raw_features, edge_raw_features, TPPR_list

def quantile_(threshold: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
    full_length = timestamps.shape[0]
    val_idx = int(threshold*full_length)

    if not isinstance(timestamps, torch.Tensor):
        timestamps = torch.from_numpy(timestamps)
    train_mask = torch.zeros_like(timestamps, dtype=bool)
    train_mask[:val_idx] = True

    val_mask = torch.zeros_like(timestamps, dtype=bool)
    val_mask[val_idx:] = True

    return train_mask, val_mask


def get_link_prediction_data4KG_Data(dataset_name: str, snapshot: int, val_ratio: float = 0.1, test_ratio: float=0.8, kg_path: Optional[dict[str, str]] = None):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    graph, idx_list = data_load(dataset=dataset_name, emb_size = 64, path=kg_path)
    node_raw_features, edge_raw_features = graph.pos

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172   
    # assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
    
    graph.pos = node_raw_features, edge_raw_features

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph.edge_attr, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    
    full_data = Data(src_node_ids=graph.edge_index[0], dst_node_ids=graph.edge_index[1], node_interact_times=graph.edge_attr, edge_ids=graph.edge_attr, labels=graph.y)
    src_node_ids = full_data.src_node_ids.astype(np.longlong)
    dst_node_ids = full_data.dst_node_ids.astype(np.longlong)
    node_interact_times = full_data.node_interact_times.astype(np.float64)
    edge_ids = full_data.edge_ids.astype(np.longlong)
    labels = full_data.labels


    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = np.isin(src_node_ids, list(new_test_node_set)) # src_node_ids.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = np.isin(dst_node_ids, list(new_test_node_set)) # dst_node_ids.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels)

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels)

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels)

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels)

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels)

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, [full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data]