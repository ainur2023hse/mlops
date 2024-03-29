import numpy as np
from sklearn.metrics import f1_score


def accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return np.mean(predicted == labels)


def f1(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return f1_score(labels, predicted)


tracked_metrics = {"accuracy": accuracy, "f1-score": f1}


def calculate_metrics(scores: list, labels: list) -> dict:
    """Compute all the metrics from tracked_metrics dict using scores and labels."""

    assert len(labels) == len(scores), print(
        "Label and score lists are of different size"
    )

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)

    metric_results = {}
    for k, v in tracked_metrics.items():
        metric_value = v(scores_array, labels_array)
        metric_results[k] = metric_value

    return metric_results
