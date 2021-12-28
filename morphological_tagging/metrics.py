from typing import Dict

import torch
import torchmetrics
from torchmetrics import MetricCollection

def clf_metrics(K: int, prefix: str, ignore_idx: int = -1):
    """Some useful multilabel, multiclass metrics to track.

    Args:
        K (int): number of classes
        prefix (str): prefix to append to logging name

    Returns:
        MetricCollection
    """
    clf_metrics = MetricCollection(
        {"acc": torchmetrics.Accuracy(num_classes=K, ignore_index=ignore_idx),
        "accweighted": torchmetrics.Accuracy(num_classes=K, average='weighted', ignore_index=ignore_idx),
        "f1micro": torchmetrics.F1(num_classes=K, average='micro', ignore_index=ignore_idx),
        "f1macro": torchmetrics.F1(num_classes=K, average='macro', ignore_index=ignore_idx)}
    )

    return clf_metrics.clone(prefix=prefix)

@torch.no_grad()
def binary_ml_clf_metrics(logits, targets, prefix: str, ignore_idx: int = -1) -> Dict[str, torch.Tensor]:
    """A function for getting useful classification metrics in the case of multi-dimensional, multi-label, but binary classification.

    Args:
        logits ([type]): [description]
        targets ([type]): [description]
        prefix (str): [description]
        ignore_idx (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """

    mask = torch.where(
        targets != ignore_idx,
        1.0,
        torch.nan)

    preds = torch.round(torch.sigmoid(logits))
    match = mask * (preds == targets).float()

    acc = torch.nanmean(match, dim=(0,1))

    fn = torch.nansum(mask * torch.logical_and(match == 0, targets == 0).float(), dim=(0,1))
    fp = torch.nansum(mask * torch.logical_and(match == 0, targets == 1).float(), dim=(0,1))
    tn = torch.nansum(mask * torch.logical_and(match == 1, targets == 0).float(), dim=(0,1))
    tp = torch.nansum(mask * torch.logical_and(match == 1, targets == 1).float(), dim=(0,1))

    prevalence = torch.sum(targets == 1, dim=(0,1)) / torch.sum(torch.logical_or(targets == 0, targets == 1), dim=(0,1))

    precision = tp / (tp + fp)
    precision[torch.isnan(precision)] = 0.0

    recall = tp / (tp + fn)
    recall[torch.isnan(recall)] = 0.0

    f1 = 2 * (precision * recall / (precision + recall))
    f1[torch.isnan(f1)] = 0.0

    return {
        f"{prefix}_accuracy_marco": torch.nanmean(acc),
        f"{prefix}_accuracy_micro": torch.sum(prevalence * acc) / torch.sum(prevalence),
        f"{prefix}_precision_macro": torch.mean(precision),
        f"{prefix}_precision_micro": torch.sum(prevalence * precision) / torch.sum(prevalence),
        f"{prefix}_recall_macro": torch.mean(recall),
        f"{prefix}_recall_micro": torch.sum(prevalence * recall) / torch.sum(prevalence),
        f"{prefix}_f1_macro": torch.mean(f1),
        f"{prefix}_f1_micro": torch.sum(prevalence * f1) / torch.sum(prevalence)
    }