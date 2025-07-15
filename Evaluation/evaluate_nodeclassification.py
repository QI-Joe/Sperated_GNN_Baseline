from doctest import FAIL_FAST
from typing import Union
import torch_geometric as tg
import torch
import torch.nn as nn
from utils.MVGRL_func import get_split
from torch_geometric.datasets import Planetoid
from typing import Optional
from utils.CLDG_func import *
import torch.nn.functional as F
from models.MVGRL_node import MVGEncoder

import torch as th
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score, recall_score, f1_score

from utils.my_dataloader import Temporal_Dataloader
"""
OK, so this is evaluation file for node classification task, which include model of GCA, MVGRL and CLDG
Class may not involved, in first stage only MVGRL evaluated method will be invovled. 
structure may refer to TGB_baseline: https://github.com/fpour/TGB_Baselines/tree/main/evaluation
"""

class LogRegression(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogRegression, self).__init__()
        self.lin = torch.nn.Linear(in_channels, num_classes)
        nn.init.xavier_uniform_(self.lin.weight.data)
        # torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.lin(x)
        return ret

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        # h = self.lin_src(z_src) + self.lin_dst(z_dst)
        # h = h.relu()
        h = F.cosine_similarity(self.lin_src(z_src), self.lin_dst(z_dst))
        return self.lin_final(h)

def get_idx_split(dataset, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')

def Simple_Regression(embedding: torch.Tensor, label: Union[torch.Tensor | np.ndarray], num_classes: int, \
                      num_epochs: int = 1500,  project_model=None, return_model: bool = False, keeper_no_train: bool=False) -> tuple[float, float, float, float]:
    device = embedding.device
    if not isinstance(label, torch.Tensor):
        label = torch.LongTensor(label).to(device)
    elif label.device != device:
        label = label.to(device)
    linear_regression = LogRegression(embedding.size(1), num_classes).to(device) if project_model==None else project_model
    f = nn.LogSoftmax(dim=-1)
    optimizer = Adam(linear_regression.parameters(), lr=0.01, weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    if keeper_no_train:
        num_epochs=0

    for epoch in range(num_epochs):
        linear_regression.train()
        optimizer.zero_grad()
        output = linear_regression(embedding)
        loss = loss_fn(f(output), label)

        loss.backward(retain_graph=False)
        optimizer.step()

        if (epoch+1) % 500 == 0:
            print(f'LogRegression | Epoch {epoch}: loss {loss.item():.4f}')

    with torch.no_grad():
        projection = linear_regression(embedding)
        y_true, y_hat = label.cpu().numpy(), projection.argmax(-1).cpu().numpy()
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_hat), \
                                        precision_score(y_true, y_hat, average='macro', zero_division=0), \
                                        recall_score(y_true, y_hat, average='macro'),\
                                        f1_score(y_true, y_hat, average='macro')
        prec_micro, recall_micro, f1_micro = precision_score(y_true, y_hat, average='micro', zero_division=0), \
                                            recall_score(y_true, y_hat, average='micro'),\
                                            f1_score(y_true, y_hat, average='micro')
    if return_model:
        return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, linear_regression
    
    return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, None


def log_regression(
    dataset: Tuple[dict[str, torch.Tensor]],  # (train, val, test, nn_val, nn_test)
    evaluator: nn.modules,
    model_name: str,
    num_classes: int,
    device: object,
    num_epochs: int = 500,
    verbose: bool = True,
    preload_split=None
):
    """
    Logistic‐regression on precomputed embeddings with:
      - adaptive evaluation interval in [10,30,50,100]
      - on any val_acc ↑ ⇒ bump to next interval
      - on 1st val_acc ↓ ⇒ interval=30
      - on 2nd val_acc ↓ ⇒ interval=50
      - on 3rd val_acc ↓ ⇒ stop training early
    """

    # unpack
    train, val, test, nn_val, nn_test = dataset
    num_classes = test["label"].unique().size(0)

    # model + loss + optimizer
    model_input = train["emb"].size(1)
    classifier  = LogRegression(model_input, num_classes).to(device)
    f           = nn.LogSoftmax(dim=-1)
    loss_fn     = nn.CrossEntropyLoss()
    optimizer   = Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-5)

    # bookkeeping
    best_train_acc     = 0.0
    best_val_acc       = 0.0
    best_test_accuracy = 0.0        # uses sklearn accuracy_score
    best_test_record   = dict()

    # adaptive‐interval state
    intervals         = [10, 30, 50, 100]
    idx               = 0
    current_interval  = intervals[idx]
    prev_val_acc      = None
    drop_count        = 0

    for epoch in range(num_epochs):
        # ——— TRAIN STEP ———
        classifier.train()
        optimizer.zero_grad()
        out = classifier(train["emb"].to(device))
        loss = loss_fn(f(out), train["label"].to(device))
        loss.backward()
        optimizer.step()

        # ——— SKIP until next eval point ———
        if (epoch + 1) % current_interval != 0:
            continue

        # ——— EVAL STEP ———
        classifier.eval()
        with torch.no_grad():
            # validation acc
            v_logits = classifier(val["emb"].to(device))
            v_preds  = v_logits.argmax(dim=-1)
            v_true   = val["label"].to(device)
            val_acc  = evaluator.eval({
                'y_true': v_true.view(-1,1),
                'y_pred': v_preds.view(-1,1)
            })['acc']

            # training acc (for logging / best‐val tracking)
            t_logits  = classifier(train["emb"].to(device))
            t_preds   = t_logits.argmax(dim=-1)
            t_true    = train["label"].to(device)
            train_acc = evaluator.eval({
                'y_true': t_true.view(-1,1),
                'y_pred': t_preds.view(-1,1)
            })['acc']

            # —— 1) ADAPTIVE‐INTERVAL LOGIC ——
            if prev_val_acc is not None:
                if val_acc > prev_val_acc:
                    # bump to next longer interval
                    if idx < len(intervals) - 1:
                        idx += 1
                        current_interval = intervals[idx]
                    drop_count = 0
                    if verbose:
                        print(f"[Epoch {epoch+1}] val ↑ {prev_val_acc:.4f}→{val_acc:.4f}, "
                              f"interval→{current_interval}")
                else:
                    # a drop
                    drop_count += 1
                    if drop_count == 1:
                        idx = 1  # force 30
                        current_interval = intervals[idx]
                        if verbose:
                            print(f"[Epoch {epoch+1}] 1st val ↓ {prev_val_acc:.4f}→{val_acc:.4f}, "
                                  f"interval→{current_interval}")
                    elif drop_count == 2:
                        idx = 2  # force 50
                        current_interval = intervals[idx]
                        if verbose:
                            print(f"[Epoch {epoch+1}] 2nd val ↓, interval→{current_interval}")
                    else:
                        if verbose:
                            print(f"[Epoch {epoch+1}] 3rd val ↓, stopping early.")
                        break

            prev_val_acc = val_acc

            # —— 2) TRACK BEST TRAIN/VAL BY BEST‐VAL RULE ——
            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                best_train_acc = train_acc

            # —— 3) EVALUATE TEST & OTHER METRICS ——
            y_true_GPU = test["label"].view(-1,1).to(device)
            y_hat_GPU  = classifier(test["emb"].to(device)).argmax(-1).view(-1,1)

            # evaluator test‐acc
            test_acc = evaluator.eval({
                'y_true': y_true_GPU,
                'y_pred': y_hat_GPU
            })['acc']

            # sklearn metrics
            y_true = y_true_GPU.cpu().numpy().ravel()
            y_hat  = y_hat_GPU.cpu().numpy().ravel()
            accuracy      = accuracy_score(y_true, y_hat)
            precision     = precision_score(y_true, y_hat, average='macro', zero_division=0)
            recall        = recall_score(y_true, y_hat, average='macro')
            f1            = f1_score(y_true, y_hat, average='macro')
            micro_prec    = precision_score(y_true, y_hat, average='micro', zero_division=0)
            micro_recall  = recall_score(y_true, y_hat, average='micro')
            micro_f1      = f1_score(y_true, y_hat, average='micro')

            # optional nn_val / nn_test
            outer_evaluate = {}
            if nn_val is not None and nn_test is not None:
                nv_t, nv_h = nn_val["label"].view(-1), classifier(nn_val["emb"]).argmax(-1)
                nt_t, nt_h = nn_test["label"].view(-1), classifier(nn_test["emb"]).argmax(-1)
                nn_val_acc    = accuracy_score(nv_t.cpu(), nv_h.cpu())
                nn_val_prec   = precision_score(nv_t.cpu(), nv_h.cpu(), average='macro', zero_division=0)
                nn_val_rec    = recall_score(nv_t.cpu(), nv_h.cpu(), average='macro')
                nn_val_f1     = f1_score(nv_t.cpu(), nv_h.cpu(), average='macro')
                nn_test_acc   = accuracy_score(nt_t.cpu(), nt_h.cpu())
                nn_test_prec  = precision_score(nt_t.cpu(), nt_h.cpu(), average='macro', zero_division=0)
                nn_test_rec   = recall_score(nt_t.cpu(), nt_h.cpu(), average='macro')
                nn_test_f1    = f1_score(nt_t.cpu(), nt_h.cpu(), average='macro')
                outer_evaluate = {
                    "nn_val_accuracy":   nn_val_acc,
                    "nn_val_precision":  nn_val_prec,
                    "nn_val_recall":     nn_val_rec,
                    "nn_val_f1":         nn_val_f1,
                    "nn_test_accuracy":  nn_test_acc,
                    "nn_test_precision": nn_test_prec,
                    "nn_test_recall":    nn_test_rec,
                    "nn_test_f1":        nn_test_f1,
                }

            # —— 4) KEEP BEST‐TEST BY sklearn ACCURACY —— 
            if accuracy > best_test_accuracy:
                best_test_accuracy = accuracy
                best_test_record   = {
                    "test_acc":    test_acc,
                    "accuracy":    accuracy,
                    "precision":   precision,
                    "recall":      recall,
                    "f1":          f1,
                    "micro_prec":  micro_prec,
                    "micro_recall":micro_recall,
                    "micro_f1":    micro_f1,
                    **outer_evaluate
                }

            # —— 5) VERBOSE LOGGING ——
            if verbose:
                print(f"(PE)|Logreg epoch {epoch+1}:"
                      f" loss={loss.item():.4f},"
                      f" val={val_acc:.4f},"
                      f" best_val={best_val_acc:.4f},"
                      f" cur_test={test_acc:.4f},"
                      f" best_test_acc={best_test_accuracy:.4f}\n"
                )

    # final return: best train/val + best test block
    return {
        "train_acc": best_train_acc,
        "val_acc":   best_val_acc,
        **best_test_record
    }


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}


def eval_GAT(emb: tuple[torch.Tensor], data: tuple[Data, Data], num_classes: int, device: str, split_ratio: float = 0.4, *args) -> dict[str, float]:
    """
    representation of data here is supposed to be a intermeidate calling method of label, which
    specified be defiened as a list, or a numpy array; but torch.Tensor is not recommended since is over-captability
    """
    
    trian_emb, test_emb = emb
    train_label, test_label = data[0].y, data[1].y
    
    train_idx, val_idx = list(range(int(0.4*trian_emb.size(0)))), list(range(int(0.4*trian_emb.size(0)), trian_emb.size(0)))
    train_emb_gen, val_emb_gen = trian_emb[train_idx], trian_emb[val_idx]

    train_label, val_label = train_label[train_idx], train_label[val_idx]

    train = {"emb": train_emb_gen, "label": train_label}
    val = {"emb": val_emb_gen, "label": val_label}
    test = {"emb": test_emb, "label": test_label}

    dataset = (train, val, test)
    return log_regression(dataset=dataset, evaluator=MulticlassEvaluator(), model_name="GAT", num_classes=num_classes, device=device, num_epochs=100)


def eval_GAT_SL(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models:nn.Linear, \
                   rb_task, train: bool, \
                   device: str="cuda:0"):
    """
    in SL trianing that the validation and text is sperated not doing it together, thus the same learning MLP should be used
    data: needed due to we need correct label
    """
    nn_res = dict()
    if train:
        train_emb = emb[data.train_mask].detach()
        train_truth = data.y[data.train_mask]
        return Simple_Regression(train_emb, train_truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=False)
    
    val_emb = emb[data.val_mask].detach()
    truth = data.y[data.val_mask].detach()
    res, model = Simple_Regression(val_emb, truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=True)
    
    if rb_task == "imbalance" or rb_task == "fsl":
        nn_emb = emb[data.nn_val_mask].detach()
        nn_truth = data.y[data.nn_val_mask].detach()
        nn_res, model = Simple_Regression(nn_emb, nn_truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=True)
        nn_res = {f"nn_{key}": value for key, value in nn_res.items()}
    return {**res, **nn_res}, model

def eval_Graphsage_SL(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models:nn.Linear, \
                   rb_task, device: str="cuda:0", train: bool=False):
    nn_res = dict()
    if train:
        train_emb = emb[data.train_mask].detach()
        train_truth = torch.from_numpy(np.array(data.y)[data.train_mask])
        return Simple_Regression(train_emb, train_truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=False)
    
    val_emb = emb[data.val_mask].detach()
    truth = torch.from_numpy(np.array(data.y)[data.val_mask])
    res, model = Simple_Regression(val_emb, truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=True)
    
    if rb_task == "imbalance" or rb_task == "fsl":
        nn_emb = emb[data.nn_val_mask].detach()
        nn_truth = torch.from_numpy(np.array(data.y)[data.nn_val_mask])
        nn_res, model = Simple_Regression(nn_emb, nn_truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=True)
        nn_res = {f"nn_{key}": value for key, value in nn_res.items()}
    return {**res, **nn_res}, model

def eval_model_Dy(emb: torch.Tensor, truth: torch.Tensor, num_classes: int, prj_model, no_train: bool=False):
    truth = truth.detach()
    return Simple_Regression(emb, truth, num_classes=num_classes, project_model=prj_model, return_model=False, keeper_no_train=True)

def eval_GCONV_SL(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models:nn.Linear, \
                   rb_task, is_train: bool, \
                   device: str="cuda:0"):
    """
    in SL trianing that the validation and text is sperated not doing it together, thus the same learning MLP should be used
    data: needed due to we need correct label
    """
    nn_res = dict()
    if is_train:
        train_emb = emb[data.train_mask].detach()
        train_truth = data.y[data.train_mask]
        return Simple_Regression(train_emb, train_truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=False)
    
    val_emb = emb[data.val_mask].detach()
    truth = data.y[data.val_mask].detach()
    res, model = Simple_Regression(val_emb, truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=True)
    
    if rb_task == "imbalance" or rb_task == "fsl":
        nn_emb = emb[data.nn_val_mask].detach()
        nn_truth = data.y[data.nn_val_mask].detach()
        nn_res, model = Simple_Regression(nn_emb, nn_truth, num_classes=num_classes, project_model=models, return_model=True, keeper_no_train=True)
        nn_res = {f"nn_{key}": value for key, value in nn_res.items()}
    return {**res, **nn_res}, model

def pre_eval_Roland(model, test_data, device):
    model.eval()
    test_data = test_data.to(device)

    h, _ = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    pred_cont = torch.sigmoid(h).cpu().detach().numpy()
    label = test_data.edge_label.cpu().detach().numpy()
    avgpr_score = average_precision_score(label, pred_cont)
    
    return avgpr_score


def eval_Roland_CL(embs: tuple[torch.Tensor], data: tuple[Data], num_classes: int, device: str, split_ratio: float = 0.4, *args) -> dict[str, float]:
    trian_emb, test_emb = embs
    train_label, test_label = data[0].y, data[1].y
    
    train_node, test_node = data[0].idx2node.index_Temporal.values, data[1].idx2node.index_Temporal.values
    train_idx, val_idx = list(range(int(0.4*len(train_node)))), list(range(int(0.4*len(train_node)), len(train_node)))
    train_emb_gen, val_emb_gen = trian_emb[train_node[train_idx]], trian_emb[train_node[val_idx]]

    train_label, val_label = train_label[train_idx], train_label[val_idx]

    train = {"emb": train_emb_gen, "label": train_label}
    val = {"emb": val_emb_gen, "label": val_label}
    test = {"emb": test_emb[test_node], "label": test_label}

    dataset = (train, val, test)
    return log_regression(dataset=dataset, evaluator=MulticlassEvaluator(), model_name="RoLAND", num_classes=num_classes, device=device, num_epochs=100)

def eval_Roland_SL(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models:nn.Linear, \
                   is_val: bool, is_test: bool, \
                   device: str="cuda:0", split_ratio: float=0.1):
    """
    in SL trianing that the validation and text is sperated not doing it together, thus the same learning MLP should be used
    data: needed due to we need correct label
    """
    if is_val and not is_test:
        emb = emb[data.val_mask].detach()
        truth = data.y[data.val_mask].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=True, num_epochs=2000)
    elif is_test and not is_val:
        ground_node_mask = data.layer2_n_id.index_Temporal.values
        test_indices = ground_node_mask[data.test_mask]
        emb = emb[test_indices].detach()
        truth = data.y[test_indices].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=False, num_epochs=2000)
    raise ValueError(f"is_val, is_test should not be the same. is_val: {is_val}, is_test: {is_test}")


def eval_CLDG(embedding_model: LogRegression,
              embeddings: tuple[th.Tensor],  
              DATASET: str, 
              trainTnum: int,
              in_label: pd.DataFrame = None,
              testIdx: Optional[th.Tensor] = None,
              device_id="cuda:0", 
              idxloader: object=None,
              *args) -> dict[str, float]:
    
    ''' Linear Evaluation '''
    # if DATASET == "dblp":
    #     labels, train_idx, val_idx, test_labels, n_classes = CLDG_testdataloader(DATASET, testIdx, idxloader=idxloader)
    if DATASET == "mathoverflow" or DATASET == "dblp":
        train_nodes, test_nodes = testIdx
        labels, test_label = idxloader.node.label[train_nodes].values, idxloader.node.label[test_nodes].values
        labels, test_labels = torch.tensor(labels), torch.tensor(test_label).to(device_id)
        lenth = len(train_nodes)
        label_train, label_val = list(range(int(0.4*lenth))), list(range(int(0.4*lenth),lenth))
        train_idx, val_idx = train_nodes[:int(0.4*lenth)], train_nodes[int(0.4*lenth):]
    else: 
        raise NotImplementedError("This kind of dataset import is not supported....")

    train_val_emb, test_emb = embeddings

    train_embs = train_val_emb[train_idx] # .to(device_id)
    val_embs = train_val_emb[val_idx]
    test_embs = test_emb[test_labels]

    n_classes = torch.unique(labels)[-1].item()+1
    label = labels.to(device_id)

    train_labels = label[label_train].clone().detach()
    val_labels = label[label_val].clone().detach()

    train = {"emb": train_embs, "label": train_labels}
    val = {"emb": val_embs, "label": val_labels}
    test = {"emb": test_embs, "label": test_labels}

    data = (train, val, test)

    return log_regression(dataset=data, evaluator=MulticlassEvaluator(), model_name="CLDG", device = device_id, num_classes=n_classes, num_epochs=100)