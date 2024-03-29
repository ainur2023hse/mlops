import numpy as np
from torch import device, no_grad
from torch.utils.data import DataLoader

from .metrics import calculate_metrics
from .model import Model


@no_grad()  # we do not need to save gradients on evaluation
def test(
    model: Model,
    batch_generator: DataLoader,
    _device: device,
) -> dict:
    """
    Evaluate the model using data from
    batch_generator and metrics defined above.
    """

    # disable dropout / use averages for batch_norm
    model.model.train(False)

    # save scores, labels and loss values for performance logging
    score_list = []
    label_list = []
    loss_list = []

    for X_batch, y_batch in batch_generator:
        logits = model.model(X_batch.to(_device))
        scores = 1 / (1 + np.exp(-logits[:, 1].detach().numpy()))
        labels = y_batch.numpy().tolist()

        # compute loss value
        loss = model.loss_function(logits, y_batch.to(_device))

        # save the necessary data
        loss_list.append(loss.detach().cpu().numpy().tolist())
        score_list.extend(scores)
        label_list.extend(labels)

    metric_results = calculate_metrics(score_list, label_list)
    metric_results["scores"] = score_list
    metric_results["labels"] = label_list
    metric_results["loss"] = loss_list

    return metric_results
