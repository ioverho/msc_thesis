import torchmetrics
from torchmetrics import MetricCollection

def clf_metrics(K: int, prefix: str):
    """Some useful multilabel, multiclass metrics to track.

    Args:
        K (int): number of classes
        prefix (str): prefix to append to logging name

    Returns:
        MetricCollection
    """
    clf_metrics = MetricCollection(
        {"acc": torchmetrics.Accuracy(num_classes=K),
        "accweighted": torchmetrics.Accuracy(num_classes=K, average='weighted'),
        "f1micro": torchmetrics.F1(num_classes=K, average='micro'),
        "f1macro": torchmetrics.F1(num_classes=K, average='macro')}
    )

    return clf_metrics.clone(prefix=prefix)