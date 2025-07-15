import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts_concrete = predicts.argmax(-1).cpu().detach().numpy()
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts, multi_class='ovr')
    accuracy = accuracy_score(y_true=labels, y_pred=predicts_concrete)
    f1 = f1_score(y_true=labels, y_pred=predicts_concrete, average='macro')
    precision = precision_score(y_true=labels, y_pred=predicts_concrete, average='macro')
    recall = recall_score(y_true=labels, y_pred=predicts_concrete, average='macro')

    return {'roc_auc': roc_auc, 'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall}
