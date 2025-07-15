import argparse
import torch
from Evaluation.time_evaluation import TimeRecord
from models.GCA_node import Encoder as GCAEncoder
from models.GCA_node import GRACE
from utils.GCA_functional import get_activation, generate_split, get_base_model, SimpleParam, number_calculate
from utils.GCA_utils import GCA_Augmentation
import random

from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting, Dynamic_Dataloader, to_cuda, str2bool
from utils.robustness_injection import Imbanlance, Few_Shot_Learning
from typing import Union, Optional

from utils.GCA_functional import *
from torch_geometric.utils import dropout_adj
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

from utils.MVGRL_func import get_split
from Evaluation.evaluate_nodeclassification import MulticlassEvaluator, log_regression
import numpy as np


def eval_GCA(model: torch.nn.Module, data: Union[Data | tuple[Data]], \
            num_classes: int, transfer, device) -> dict[str, float]:
    model.eval()
    if isinstance(data, (Data, Temporal_Dataloader)):
        t_data = data
        length = t_data.x.size(0)
        with torch.no_grad():
            z = model(t_data.x, t_data.edge_index)
        dataset = get_split(num_samples=length, emb = z, data = t_data, transfer=transfer, Nontemproal=True)
    else:
        t_data, t1_data = data
        length = t_data.x.size(0)
        with torch.no_grad():
            z = model(t_data.x, t_data.edge_index)
            z1 = model(t1_data.x, t1_data.edge_index)
        dataset = get_split(num_samples=length, emb = (z, z1), data = (t_data, t1_data), transfer=transfer, train_ratio=0.8)

    evaluator = MulticlassEvaluator()
    macro = log_regression(dataset=dataset, evaluator=evaluator, model_name="GCA", num_epochs=1000,\
                            num_classes=num_classes, device=device, preload_split=None)

    return macro



def train_GCA(model: GRACE, optimizer, param, data, feature_weights, drop_weights, args):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])
    
    if(args.graph_size == 0):
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
        loss.backward()
        optimizer.step()
        return loss.item(), z1.shape[0]
    else:
        total_loss = 0
        batch_size = 2000
        dz1, dz2 = Data(x = x_1, edge_index = edge_index_1), Data(x = x_2, edge_index = edge_index_2)
        neighbor1 = NeighborLoader(dz1, batch_size=batch_size, num_neighbors=[-1], shuffle=False)
        for idx, batch in enumerate(neighbor1):
            inter_batch_size = batch.batch_size
            seed_node = batch.n_id[:inter_batch_size]
            neighbor2 = NeighborLoader(dz2, batch_size=inter_batch_size, num_neighbors=[-1], input_nodes=seed_node, shuffle=False)
            batch2 = next(iter(neighbor2))
            assert batch.n_id[:inter_batch_size].tolist() == batch2.n_id[:inter_batch_size].tolist(), "sorry, this method seems a little problem."
            z1 = model(batch.x, batch.edge_index)[: inter_batch_size]
            z2 = model(batch2.x, batch2.edge_index)[:inter_batch_size]
            loss = model.loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/idx, batch_size


def main_GCA(outside_args, default_param, time_: TimeRecord): # done
    args = outside_args
    time_.get_dataset(args.dataset)
    time_.set_up_logger(name="time_logger")
    time_.set_up_logger()
    time_.record_start()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')
    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    dynamic = args.dynamic
    random.seed(2024)
    torch.manual_seed(2024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    rb_task = args.rb_task
    ratio = args.ratio
    epoch_interval = 1
    
    wargs = {"rb_task": rb_task, "ratio": ratio}
    graph, idxloader = data_load(args.dataset, **wargs)
    snapshot = args.snapshots
    if dynamic:
        num_classes = len(idxloader)+1
    else:
        num_classes = graph.y.max().item() + 1
    
    graph_list = Temporal_Splitting(graph, dynamic=dynamic, idxloader=idxloader).temporal_splitting(time_mode="view", snapshot=snapshot, views=snapshot-2)
    temporaLoader = Dynamic_Dataloader(graph_list, graph=graph)

    encoder = GCAEncoder(graph.pos.size(1), param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    datawarehouse: list[tuple] = []

    for t in range(3):
        time_.temporal_record()
        data = temporaLoader.get_temporal()
        data = to_cuda(data)
        data.edge_index = data.edge_index.type(torch.int64)
        
        drop_weights, feature_weights = GCA_Augmentation(data=data, param=param, args=args, device=device)
        # feature_weights somehow will return Nan value, currently will detect and switch it to 0
        # also, it will be print out
        if torch.isnan(feature_weights).any():
            val, freq = torch.unique(feature_weights, return_counts=True)
            val, freq = number_calculate(val, freq)
            print("#"+"-"*30+\
                  "\n"+\
                f'Feature weights has NaN value, will be replaced by 0')
            print(f"torch unique shows: {val}, {freq}")
            print("#"+"-"*30)
            feature_weights[torch.isnan(feature_weights)] = 0

        log = args.verbose.split(',')

        microList = []
        
        transfer = None
        if rb_task!= None or rb_task!="None":
            if rb_task == "imbalance":
                transfer = Imbanlance(ratio, train_ratio=0.8)
            elif rb_task == "fsl":
                ratio = int(ratio)
                transfer = Few_Shot_Learning(ratio)
        
        for epoch in range(1, args.num_epoches + 1):
            time_.epoch_record()

            loss, batch_size = train_GCA(model, optimizer, param, data, feature_weights, drop_weights, args)
            if 'train' in log:
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, node Num={data.num_nodes}')
            
            time_.epoch_end(batch_size)
            if (epoch+1) % epoch_interval == 0:
                all_data = (data, to_cuda(temporaLoader.get_T1graph(t)))
                micro = eval_GCA(model=model, data=all_data, device=device, transfer=transfer, num_classes=num_classes)

                if 'eval' in log:
                    print(f'(E) | Epoch={epoch:04d}, avg_acc = {micro["test_acc"]:03f}')
                microList.append(micro)
        if rb_task == "imbalance" or rb_task == "fsl":
            train_acc, val_acc, test_acc, accuracy, precision, recall, f1, \
                micro_prec, micro_recall, micro_f1, \
                nn_val_accuracy, nn_val_precision, nn_val_recall, nn_val_f1, \
                nn_test_accuracy, nn_test_precision, nn_test_recall, nn_test_f1 \
                = zip(*[list(data.values()) for data in microList])
        else:
            train_acc, val_acc, test_acc, accuracy, precision, recall, f1, \
                micro_prec, micro_recall, micro_f1 \
                = zip(*[list(data.values()) for data in microList])
        time_.score_record(microList, data.num_nodes, t)
        # train_acc, val_acc .etc. shape:
        # based on how many times test, if every 50 epoch test once, then it will be epoch/50 length
        # thus, last acc should be focused since it will be the highest one.

        final_micro = eval_GCA(model=model, data=all_data, device=device, transfer=transfer, num_classes=num_classes)
        if rb_task == "imbalance" or rb_task == "fsl":
            datawarehouse.append([
                final_micro["test_acc"], train_acc, val_acc, test_acc, accuracy, precision, recall, f1,
                (nn_val_accuracy, nn_val_precision, nn_val_recall, nn_val_f1,
                nn_test_accuracy, nn_test_precision, nn_test_recall, nn_test_f1)
            ])
        else:
            datawarehouse.append([final_micro["test_acc"], train_acc, val_acc, test_acc, accuracy, precision, recall, f1, None])
        
        temporaLoader.update_event(t)
        if 'final' in log:
            print(f'{final_micro}')
        
        time_.temporal_end(data.num_nodes)

    time_.record_end()
    time_.to_log()
    for i in range(len(datawarehouse)):
        final_micro_test_acc, train_acc, val_acc, test_acc, accuracy, precision, recall, f1, nn_task = datawarehouse[i]
        print(f'View {i}, \n \
            Final Test Acc {final_micro_test_acc:04f}, \n \
            Train Acc {np.mean(train_acc):05f}, \n \
            Test Acc {np.mean(test_acc):05f}, \n \
            Val Acc {np.mean(val_acc):05f} \n\n \
            Avg accuracy {np.mean(accuracy):05f}, \n \
            Avg precision {np.mean(precision):05f}, \n \
            Avg recall {np.mean(recall):05f}, \n \
            Avg f1 {np.mean(f1):05f}, \n\n ')
        if nn_task != None:    
            (nn_val_accuracy, nn_val_precision, nn_val_recall, nn_val_f1,
         nn_test_accuracy, nn_test_precision, nn_test_recall, nn_test_f1) = nn_task
            print(f'NN Val Accuracy {np.mean(nn_val_accuracy):05f}, \n \
            NN Val Precision {np.mean(nn_val_precision):05f}, \n \
            NN Val Recall {np.mean(nn_val_recall):05f}, \n \
            NN Val F1 {np.mean(nn_val_f1):05f}, \n\n \
            NN Test Accuracy {np.mean(nn_test_accuracy):05f}, \n \
            NN Test Precision {np.mean(nn_test_precision):05f}, \n \
            NN Test Recall {np.mean(nn_test_recall):05f}, \n \
            NN Test F1 {np.mean(nn_test_f1):05f}')


def GCA_config(model_detail):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument("--snapshots", type=int, default=20)
    parser.add_argument('--extra_abondan', type=int, default=0)
    parser.add_argument("--num_epoches", type=int, default=2)
    parser.add_argument('--param', type=str, default='local:coauthor_cs.json')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--graph_size', type=int, default=0)
    parser.add_argument("--dynamic", type=str2bool, default=False)
    parser.add_argument("--rb_task", type=str, default="imbalance") # edge_disturb
    parser.add_argument("--ratio", type=float, default=0.1)
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
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args(model_detail)
    return args, default_param



def main(extra, args):
    global default_param
    sid = args.special_id
    time_rec = TimeRecord(model_name="GCA", sid=sid)

    gca_config, default_param = GCA_config(extra)
    main_GCA(gca_config, default_param, time_rec)