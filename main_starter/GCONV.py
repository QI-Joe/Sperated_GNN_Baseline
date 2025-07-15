import argparse
import torch
from Evaluation.evaluate_nodeclassification import LogRegression, eval_GCONV_SL, eval_model_Dy
from Evaluation.time_evaluation import TimeRecord

from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting, Dynamic_Dataloader, to_cuda
from utils.robustness_injection import Imbanlance, Few_Shot_Learning

import numpy as np
import torch.nn as thnn
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data


import numpy as np
from itertools import chain
from models.GCONV import Benchmark_GCONV
import random
import warnings

# Ignore specific warning by message
warnings.filterwarnings("ignore")
MODEL = "gonv"

# :author marks: time-reocrding system not involved, and t, t+1 time not import

def main_GCONV(args):
    global score_, present_
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset
    hidd_dim1, hidden_dim2 = args.hidden_dim1, args.hidden_dim2
    output_dim = args.output_dim
    train_epoch = args.train_epoch
    snapshot = args.snapshot
    dynamic = args.dynamic
    non_split = True
    epoch_interval = 50
    rb_task = args.rb_task
    ratio = args.ratio
    view = snapshot - 2
    
    wargs = {"rb_task": rb_task, "ratio": ratio}

    score_.get_dataset(dataset_name)
    present_.get_dataset(dataset_name)
    score_.set_up_logger()
    present_.set_up_logger("time")
    present_.record_start()

    data, idxloader = data_load(dataset_name, **wargs)
    num_classes = data.y.max().item() + 1

    dataloader = Temporal_Splitting(data, dynamic=dynamic, idxloader=idxloader).temporal_splitting(time_mode="view", snapshot=snapshot, \
                                        views=snapshot-2, strategy = "sequencial", non_split=non_split)
    data_neightbor = Dynamic_Dataloader(dataloader, graph=data)

    model = Benchmark_GCONV(input_dim=data.pos.size(1), output_dim = output_dim, num_nodes = data.num_nodes, device = device, \
                             hidden_dim1=hidd_dim1, hidden_dim2=hidden_dim2).to(device)
    projector = LogRegression(output_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(chain(model.parameters(), projector.parameters()), lr=2e-3, weight_decay=1e-5)

    loss_fn = thnn.CrossEntropyLoss()
    random.seed(2024)
    torch.manual_seed(2024)
    goal_list=list()

    for t in range(view):
        present_.temporal_record()
        temporal = data_neightbor.get_temporal()
        num_nodes = temporal.num_nodes
        temporal = to_cuda(temporal)
        
        transform = RandomNodeSplit(num_val=0.2, num_test=0.0)
        if rb_task!= None or rb_task!="None":
            if rb_task == "imbalance":
                transform = Imbanlance(ratio, train_ratio=0.8)
            elif rb_task == "fsl":
                ratio = int(ratio)
                transform = Few_Shot_Learning(ratio)
            
        temporal = transform(temporal)
        temporal_recorder, nn_test_metrics = list(), dict()

        for epoch in range(train_epoch):
            present_.epoch_record()
            model.train()
            projector.train()
            optimizer.zero_grad()

            node_embedding = model(temporal.x, temporal.edge_index)
            node_pred = projector(node_embedding)
            loss = loss_fn(node_pred[temporal.train_mask], temporal.y[temporal.train_mask])

            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, | loss {loss.item()}")
        
            present_.epoch_end(batch_size=num_nodes)
            if not (epoch+1) % epoch_interval:
                model.eval()
                projector.eval()

                with torch.no_grad():
                    node_embedding = model(temporal.x, temporal.edge_index)
                node_pred = projector(node_embedding)
                acc = node_pred[temporal.val_mask].max(1)[1].eq(temporal.y[temporal.val_mask]).sum().item() / temporal.val_mask.sum().item()
                train_metics, decoder = eval_GCONV_SL(emb=node_embedding, data = temporal, num_classes=num_classes, models=None, rb_task=rb_task, is_train=True)
                
                print("Train Acc is:", acc)
                val_metrics, decoder = eval_GCONV_SL(emb=node_embedding, data = temporal, num_classes=num_classes, models=decoder, rb_task=rb_task, is_train=False)

                t1_temporal_graph = data_neightbor.get_T1graph(t)
                t1_temporal_graph = to_cuda(t1_temporal_graph)              
                t1_emb = model(t1_temporal_graph.x, t1_temporal_graph.edge_index)
                test_metrics, _ = eval_model_Dy(t1_emb, t1_temporal_graph.y, num_classes, prj_model=decoder)
                
                if rb_task == "imbalance":
                    t1_temporal_graph = transform.test_processing(t1_temporal_graph)
                    nn_test_emb, nn_test_label = t1_emb[t1_temporal_graph.nn_test_mask], t1_temporal_graph.y[t1_temporal_graph.nn_test_mask]
                    nn_test_metrics, _ = eval_model_Dy(emb=nn_test_emb, truth=nn_test_label, num_classes=num_classes, prj_model=decoder)
                    nn_test_metrics = {f"nn_{key}": value for key, value in nn_test_metrics.items()}

                test_metrics["train_acc"], test_metrics["val_acc"] = acc, val_metrics["accuracy"]
                test_metrics = {**test_metrics, **nn_test_metrics}
                print(f"Train Acc: {acc:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, NN Test Acc: {nn_test_metrics.get('nn_accuracy', 0):.4f}")
                temporal_recorder.append(test_metrics)

        goal_list.append([temporal.num_nodes, temporal_recorder])

        present_.temporal_end(temporal.num_nodes)
        present_.score_record(temporal_recorder, temporal.num_nodes, t)
        data_neightbor.update_event(t)
    present_.record_end()
    present_.to_log()
    score_.record_end()
    score_.fast_processing([i[1] for i in goal_list])

    for i in range(len(goal_list)):
        num_nodes, temporal_data = goal_list[i]
        last_choice = temporal_data[-1]
    
        print(f"View {t+1}, \nInclude {num_nodes:05d} nodes, \n")
        for key, value in last_choice.items():
            print(f"{key}: {value:.5f}")

def GCONV_config(model_detail):
    parser = argparse.ArgumentParser(description='GCONV Configuration')
    parser.add_argument('--hidden_dim1', type=int, default=64, help='Hidden dimension 1')
    parser.add_argument('--hidden_dim2', type=int, default=32, help='Hidden dimension 2')
    parser.add_argument('--output_dim', type=int, default=16, help='Output dimension')
    parser.add_argument('--train_epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--snapshot', type=int, default=20, help='Snapshot interval')
    parser.add_argument('--dataset', type=str, default='dblp', help='Dataset name')
    parser.add_argument('--dynamic', type=bool, default=False)
    parser.add_argument("--rb_task", type=str, default="edge_disturb") # edge_disturb
    parser.add_argument("--ratio", type=float, default=0.5)
    args = parser.parse_args(model_detail)

    return args


score_, present_ = None, None
def main(extra, args):
    special_id = args.special_id
    global score_, present_
    score_ = TimeRecord(MODEL, special_id)
    present_ = TimeRecord(MODEL, special_id)
    gcn_config = GCONV_config(extra)
    main_GCONV(gcn_config)
    