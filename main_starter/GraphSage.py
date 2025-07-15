import argparse
import torch
from Evaluation.evaluate_nodeclassification import Simple_Regression, eval_Graphsage_SL
from Evaluation.time_evaluation import TimeRecord
import random
from typing import Union
from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting, Dynamic_Dataloader, to_cuda
from utils.robustness_injection import Imbanlance, Few_Shot_Learning

from models.GraphSage import MeanAggregator, Encoder, SupervisedGraphSage, adjacent_list_building
from torch.autograd import Variable
import numpy as np
import torch.nn as thnn
from torch_geometric.transforms import RandomNodeSplit
import copy



def eval_model_Dy_1(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, device: str="cuda:0"):
    t1_nodes = torch.tensor(data.my_n_id.node["index"].values).to(device)
    emb = emb.detach()[t1_nodes]
    truth = data.y.detach()
    return Simple_Regression(emb, truth, num_classes=num_classes, project_model=None, return_model=False)

def eval_model_Dy(emb: torch.Tensor, truth: torch.Tensor, num_classes: int, prj_model, no_train: bool=False):
    truth = truth.detach()
    return Simple_Regression(emb, truth, num_classes=num_classes, project_model=prj_model, return_model=False, keeper_no_train=no_train)



def main_GraphSage(args, time_: TimeRecord): # done
    r"""
    initially tested by dblp and mathoverflow dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument("--output_embedding_dim", type=int, default=128)
    parser.add_argument("--sample_batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dynamic", type=bool, default=False)
    parser.add_argument("--snapshot", type=int, default=20)
    parser.add_argument("--rb_task", type=str, default="edge_disturb")
    parser.add_argument("--ratio", type=float, default=0.5)
    args = parser.parse_args(args)
    time_.get_dataset(args.dataset)
    dataset, output_embedding_dim = args.dataset, args.output_embedding_dim
    time_.set_up_logger(name="time_logger")
    time_.set_up_logger()
    time_.record_start()

    snapshot = args.snapshot
    non_split = True
    dynamic = args.dynamic
    rb_task = args.rb_task
    ratio = args.ratio
    view = snapshot-2
    epoch_interival = 10

    wargs = {"rb_task": rb_task, "ratio": ratio}
    graph, idxloader = data_load(dataset, **wargs)

    if dynamic:
        num_classes = len(idxloader)
    else: 
        num_classes = graph.y.max().item() + 1

    graph_list = Temporal_Splitting(graph, dynamic=dynamic, idxloader=idxloader).temporal_splitting(time_mode="view", \
                    snapshot=snapshot, views=snapshot-2, strategy="sequential", non_split=non_split)
    dataneighbor = Dynamic_Dataloader(graph_list, graph=graph)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_storage, transform = [], None

    # over predicting?
    graphsage = SupervisedGraphSage()
    random.seed(2024)
    torch.manual_seed(2024)

    for t in range(view):
        time_.temporal_record()
        data = dataneighbor.get_temporal()
        num_nodes = data.num_nodes
        feat_data, labels = data.pos, data.y # <- porblem at here, man

        adj_lists, features = None, None
        if graphsage.cached_exam():
            adj_lists, features = graphsage.load_cache()
            # print("Cache loaded, in memored shape",features.weight.shape)
        else:
            adj_lists = adjacent_list_building(data)
            feat_dim = feat_data.shape[1]
            features = thnn.Embedding(num_nodes, feat_dim)
            features.weight = thnn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
            features = features.to(device)
            # features = features.to(device)

        agg1 = MeanAggregator(features, to_cuda=True)
        enc1 = Encoder(features, feat_dim, output_embedding_dim, adj_lists, agg1, gcn=True, to_cuda=True)
        agg2 = MeanAggregator(lambda nodes : enc1.forward(nodes).t(), to_cuda=True)
        enc2 = Encoder(lambda nodes : enc1.forward(nodes).t(), enc1.embed_dim, output_embedding_dim, adj_lists, agg2,
                base_model=enc1, gcn=True, to_cuda=True)
        enc1.num_samples = 5
        enc2.num_samples = 5

        graphsage.enc2_load(num_classes=num_classes, enc=enc2)
        
        transform = RandomNodeSplit(num_val=0.2, num_test=0.0)
        if rb_task!= None or rb_task!="None":
            if rb_task == "imbalance":
                transform = Imbanlance(ratio, train_ratio=0.8)
            elif rb_task == "fsl":
                ratio = int(ratio)
                transform = Few_Shot_Learning(ratio)
            
        data: Temporal_Dataloader = transform(data)
        train_mask, val_mask = data.train_mask, data.val_mask
        node_idx = data.my_n_id.node["index"].values
        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)

        temporal_recorder, nn_test_metrics = [], dict()
        for batch in range(args.epochs):
            time_.epoch_record()
            graphsage.train()
            optimizer.zero_grad()
            loss = graphsage.loss(node_idx[train_mask], 
                    Variable(torch.LongTensor(np.array(labels)[train_mask])).to(device)) # need to on GPU if requried
            loss.backward()
            optimizer.step()
            time_.epoch_end(args.sample_batch)
            print(f"Epoch {batch+1}, Loss: {loss.item():.4f}")

            if (batch+1) % epoch_interival ==0:
                graphsage.eval()

                train_output = graphsage.forward(node_idx).detach()
                # train_label = np.array(labels)[train_mask]
                train_eval, decoder = eval_Graphsage_SL(emb = train_output, data = data, num_classes=num_classes, models=None, rb_task=rb_task, \
                                  device=device, train = True)

                val_output = graphsage.forward(node_idx).detach()
                # val_label = np.array(labels)[val_mask]
                val_eval, decoder = eval_Graphsage_SL(emb = val_output, data = data, num_classes=num_classes, models=decoder, rb_task=rb_task, \
                                  device=device, train=False)
                
                # prevent data be mofied in Temporal Splitter
                t1_test = copy.deepcopy(dataneighbor.get_T1graph(t))

                # graphsage memory bank reset, all model should equipped with a parameter reset function
                t1_adj_list = adjacent_list_building(t1_test)
                graphsage.encoder_adj_redress(t1_test.pos, t1_adj_list)

                test_nodes = t1_test.my_n_id.node["index"].values
                test_output = graphsage.forward(test_nodes).detach().to(device)
                
                t1_test = to_cuda(t1_test, device=device)
                test_eval, _ = eval_model_Dy(emb=test_output, truth=t1_test.y, num_classes=num_classes, no_train=True, prj_model=decoder)
                
                if rb_task == "imbalance":
                    t1_test = transform.test_processing(t1_test)
                    nn_test_emb, nn_test_label = test_output[t1_test.nn_test_mask], t1_test.y[t1_test.nn_test_mask]
                    nn_test_metrics, _ = eval_model_Dy(emb=nn_test_emb, truth=nn_test_label, num_classes=num_classes, no_train=True, prj_model=decoder)
                    nn_test_metrics = {f"nn_{key}": value for key, value in nn_test_metrics.items()}

                test_eval["train_acc"], test_eval["val_acc"] = train_eval['accuracy'], val_eval['accuracy']
                test_eval = {**test_eval, **nn_test_metrics}
                temporal_recorder.append(test_eval)
                print("Train Acc is {:4f}; Validation acc {:4f}".format(train_eval["accuracy"], val_eval["accuracy"]))
        
                graphsage.t_moment_redress()
                del t1_test

        data_storage.append([num_nodes, temporal_recorder])
        print("Val_acc is: {:4f}".format(round(val_eval["accuracy"], 4)))

        time_.temporal_end(num_nodes)
        time_.score_record(temporal_recorder, num_nodes, t)

        dataneighbor.update_event(t)

    time_.record_end()
    time_.to_log()
    for i in range(len(data_storage)):
        num_nodes, temporal_recorder = data_storage[i]
        temporal_recorder = temporal_recorder[-1]
        print(f"View {i}, \n Include {num_nodes:05d} nodes, \n" )
        for key, value in temporal_recorder.items():
            print(f"{key}: {value:.5f}")
        print("\n\n")
    # suppose to be a data clean code


def main(extra, args):
    sid = args.special_id
    time_rec = TimeRecord("graphsage", sid)
    main_GraphSage(extra, time_rec)
    