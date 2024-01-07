import time

import mlflow
import numpy as np
from torch import device
from torch import save as t_save
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import Model
from .test import test


def train(
    model: Model,
    _device: device,
    train_batch_generator: DataLoader,
    val_batch_generator: DataLoader,
    ckpt_name: str,
    n_epochs: int,
) -> None:
    """Run training"""
    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    top_val_accuracy = 0
    for _epoch in range(n_epochs):
        model.model.train(True)
        for X_batch, y_batch in tqdm(train_batch_generator, desc="Training", leave=False):
            data_batch = {"img": X_batch.to(_device), "label": y_batch.to(_device)}
            model.optimizer.zero_grad()
            model.compute_loss(data_batch)
            model.loss.backward()
            model.optimizer.step()
            train_loss.append(model.loss.detach().cpu().numpy())

        # Evaluation phase
        metric_results = test(model, val_batch_generator, _device)

        # Logging
        val_loss_value = np.mean(metric_results["loss"])
        val_loss_idx.append(len(train_loss))
        val_loss.append(val_loss_value)

        val_accuracy_value = metric_results["accuracy"]
        val_f1_value = metric_results["f1-score"]

        mlflow.log_metric("epoch train loss", np.mean(train_loss))
        mlflow.log_metric("epoch val loss", val_loss_value)
        mlflow.log_metric("epoch val accuracy", val_accuracy_value)
        mlflow.log_metric("epoch val f1-score", val_f1_value)

        if val_accuracy_value > top_val_accuracy:
            top_val_accuracy = val_accuracy_value
            with open(ckpt_name, "wb") as f:
                t_save(model.model, f)

        time.sleep(30)
