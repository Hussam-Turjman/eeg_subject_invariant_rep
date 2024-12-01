from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
# import torch
from easydict import EasyDict
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, recall_score, precision_score,
                             average_precision_score, balanced_accuracy_score, confusion_matrix)
from sklearn.utils.multiclass import unique_labels


def is_conf_optimal(conf_mat: np.ndarray) -> bool:
    a = conf_mat
    a: np.ndarray
    assert a.ndim == 2
    rows = conf_mat.shape[0]
    cols = conf_mat.shape[1]
    assert rows == cols
    count_maxima = 0
    for i, j in zip(np.arange(rows), np.arange(rows)):

        column = a[:, j]
        row = a[i, :]
        column_max_idx = np.argmax(column)
        row_max_idx = np.argmax(row)
        if column_max_idx == i and row_max_idx == j:
            count_maxima += 1
    return count_maxima == rows


# def compute_conf_mat(predictions, targets, num_classes, normalize="all"):
#     task = "multiclass" if num_classes > 2 else "binary"
#     conf = ConfusionMatrix(task=task,
#                            num_classes=num_classes,
#                            normalize=normalize)
#     if isinstance(predictions, torch.Tensor):
#         res = conf(predictions, targets)
#     else:
#         res = conf(torch.from_numpy(predictions), torch.from_numpy(targets))
#     return res


def calc_conf_matrix(target, prediction, normalize=False):
    labels = unique_labels(target, prediction)
    cm = confusion_matrix(target, prediction, labels=labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    return cm, labels


def conf_matrix(target, prediction, normalize=False, title="Confusion Matrix", ax=None):
    cm, labels = calc_conf_matrix(target, prediction, normalize)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    ax = sns.heatmap(df, annot=True, vmin=0, vmax=100, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True Label')
    if title is not None:
        ax.set_title(title,fontsize=20)
    return cm


def single_value_metrics(predictions, predictions_logits, targets, targets_logits, num_classes,
                         multi_class_average="macro") -> EasyDict:
    roc = roc_auc_score(y_true=targets_logits, y_score=predictions_logits,
                        multi_class="ovr" if num_classes > 2 else "raise")
    accuracy = accuracy_score(y_true=targets, y_pred=predictions)
    balanced_accuracy = balanced_accuracy_score(y_true=targets, y_pred=predictions)
    f1 = f1_score(y_true=targets, y_pred=predictions, average="binary" if num_classes == 2 else multi_class_average)
    recall = recall_score(y_true=targets, y_pred=predictions,
                          average="binary" if num_classes == 2 else multi_class_average)
    precision = precision_score(y_true=targets, y_pred=predictions,
                                average="binary" if num_classes == 2 else multi_class_average,
                                zero_division=0.0)
    average_precision = average_precision_score(y_true=targets_logits, y_score=predictions_logits,
                                                average="macro" if num_classes == 2 else "weighted")

    return EasyDict({
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
        "average_precision": average_precision,
        "roc": roc
    })


__all__ = ["single_value_metrics", "is_conf_optimal", "conf_matrix"]
