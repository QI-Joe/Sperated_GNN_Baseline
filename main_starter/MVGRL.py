import GCL.losses as L
import GCL.augmentors as A
import argparse
import torch
from models.MVGRL_node import MVGEncoder, Encoder_Neighborloader
from models.MVGRL_node import GConv
from Evaluation.evaluate_nodeclassification import log_regression, MulticlassEvaluator
from Evaluation.time_evaluation import TimeRecord
from GCL.models import DualBranchContrast
import random
from typing import Union
from torch.optim import Adam
from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting, Dynamic_Dataloader, to_cuda, str2bool
from utils.robustness_injection import Imbanlance, Few_Shot_Learning
import time
from torch_geometric.data import Data
from utils.MVGRL_func import get_split


def eval_MVGRL(encoder_model: MVGEncoder, data: tuple[Temporal_Dataloader, Temporal_Dataloader], device, transfer) -> dict[str, float]:
    encoder_model.eval()
    num_epoch = 1000
    if not isinstance(data, (Data, Temporal_Dataloader)):
        trian_val, test = data
        z1, z2, _, _, _, _, n_id, _ = encoder_model(trian_val.x, trian_val.edge_index)
        t1 = (z1 + z2).detach()
        encoder_model.switch_mode(in_eval=True)
        tz1, tz2, _, _, _, _, n_id, _ = encoder_model(test.x, test.edge_index)
        t2 = (tz1 + tz2).detach()

        trian_test = get_split(num_samples=t1.size()[0], emb=(t1,t2), data=data,)

    else:
        z1, z2, _, _, _, _, n_id, _ = encoder_model(data.x, data.edge_index)
        t1 = (z1 + z2).detach()
        data.y = data.y[n_id]

        trian_test = get_split(num_samples=t1.size(0), emb = t1, data = data, Nontemproal=True, transfer=transfer)

    result = log_regression(dataset=trian_test, evaluator = MulticlassEvaluator(), \
                            model_name="MVGRL", num_classes=None, device=device, num_epochs=num_epoch)
    return result

def train_MVGRL(encoder_model: Union[MVGEncoder|Encoder_Neighborloader], contrast_model, data: Temporal_Dataloader, optimizer: Adam):
    encoder_model.train()
    optimizer.zero_grad()
    batch_size = data.x.shape[0]
    z1, z2, g1, g2, z1n, z2n, _, batch_size = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item(), batch_size

def main_MVGRL(unknow_parms, time_: TimeRecord): # not done
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--snapshots', type=int, default=20)
    parser.add_argument('--extra_abondan', type=int, default=0)
    parser.add_argument("--graph_size", type=int, default=1)
    parser.add_argument('--dynamic', type=str2bool, default=False)
    parser.add_argument("--rb_task", type=str, default="edge_disturb") # edge_disturb
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args(unknow_parms)

    time_.get_dataset(args.dataset)
    time_.set_up_logger(name="time_logger")
    time_.set_up_logger()
    time_.record_start()

    snapshot = args.snapshots
    dynamic = args.dynamic
    view = snapshot - 2 - args.extra_abondan
    rb_task = args.rb_task
    ratio = args.ratio
    
    wargs = {"rb_task": rb_task, "ratio": ratio}

    graph, idxloader = data_load(args.dataset, dynamic=dynamic, **wargs)
    graph_list = Temporal_Splitting(graph, dynamic=dynamic, idxloader=idxloader).temporal_splitting(time_mode="view", snapshot=snapshot, views=snapshot-2)
    dataneighbor = Dynamic_Dataloader(graph_list, graph=graph)


    aug1 = A.Identity()
    gconv1 = GConv(input_dim=graph.pos.size(1), hidden_dim=512, num_layers=2).to(device)
    gconv2 = GConv(input_dim=graph.pos.size(1), hidden_dim=512, num_layers=2).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)


    datacollector: list[list] = []

    for t in range(view):
        time_.temporal_record()
        sum_loss: list = []
        batch = dataneighbor.get_temporal()
        aug2 = A.PPRDiffusion(alpha=0.2)
        batch = to_cuda(batch)
        batch.edge_index = batch.edge_index.type(torch.LongTensor).to(device)

        if args.graph_size == 0:
            encoder_model = MVGEncoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
        else: 
            encoder_model = Encoder_Neighborloader(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
        optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)

        transfer = None
        if rb_task!= None or rb_task!="None":
            if rb_task == "imbalance":
                transfer = Imbanlance(ratio, train_ratio=0.8)
            elif rb_task == "fsl":
                ratio = int(ratio)
                transfer = Few_Shot_Learning(ratio)

        for epoch in range(1, args.num_epoches): # 2000 / 1500+
            time_.epoch_record()
            loss, bc_size = train_MVGRL(encoder_model, contrast_model, batch, optimizer)
            sum_loss.append(loss)
            print(f'(T): Epoch={epoch:03d}, loss={loss:.4f}, node Num={batch.x.shape[0]}')

            time_.epoch_end(bc_size)
            if (epoch+1) % 50 == 0:
                dataset = (batch, to_cuda(dataneighbor.get_T1graph(t)))
                test_result = eval_MVGRL(encoder_model, dataset, device=device, transfer=transfer)
                train_acc, val_acc, test_acc = test_result["train_acc"], test_result["val_acc"], test_result["test_acc"]
                print(f'(E): Epoch={epoch:03d}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}')
                datacollector.append(test_result)

        dataset = (batch, to_cuda(dataneighbor.get_T1graph(t)))
        test_result = eval_MVGRL(encoder_model, batch, device=device, transfer=transfer)

        test_result["min_loss"] = min(sum_loss)
        time_.score_record(datacollector, batch.x.shape[0], t)

        time_.temporal_end(batch.x.shape[0])
        print(f'(E): Best test F1Mi={test_result["test_acc"]:.4f}, F1Ma={test_result["precision"]:.4f}')
        dataneighbor.update_event(t)
    
    time_.record_end()
    time_.to_log()



def main(extra, args):
    model, sid = "MVGRL", args.special_id
    time_rec = TimeRecord(model_name=model, sid=sid)
    main_MVGRL(extra, time_rec)

    