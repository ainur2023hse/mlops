import time

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import save as t_save
from tqdm.auto import tqdm

from .config import device, settings
from .metrics import get_score_distributions
from .model import compute_loss, model
from .test import test_model


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_model(
    model,
    train_batch_generator,
    val_batch_generator,
    opt,
    ckpt_name=None,
    n_epochs=settings.EPOCH_NUM,
    visualize: bool = True,
):
    """Run training:"""

    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    top_val_accuracy = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train phase
        model.train(True)  # enable dropout / batch_norm training behavior
        for X_batch, y_batch in tqdm(train_batch_generator, desc="Training", leave=False):
            # move data to target device
            data_batch = {"img": X_batch.to(device), "label": y_batch.to(device)}

            # train on batch:
            optimizer.zero_grad()
            loss, model = compute_loss(model, data_batch)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().cpu().numpy())

        # Evaluation phase
        metric_results = test_model(model, val_batch_generator, subset_name="val")
        metric_results = get_score_distributions(metric_results)

        # Logging
        val_loss_value = np.mean(metric_results["loss"])
        val_loss_idx.append(len(train_loss))
        val_loss.append(val_loss_value)

        if visualize:
            # tensorboard for the poor
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(121)

            ax1.plot(train_loss, color="b", label="train")
            ax1.plot(val_loss_idx, val_loss, color="c", label="val")
            ax1.legend()
            ax1.set_title("Train/val loss.")

            fig.savefig(f"{epoch}.png")

        print(
            "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time
            )
        )
        val_accuracy_value = metric_results["accuracy"]
        if val_accuracy_value > top_val_accuracy and ckpt_name is not None:
            top_val_accuracy = val_accuracy_value

            # save checkpoint of the best model to disk
            with open(ckpt_name, "wb") as f:
                t_save(model, f)

        time.sleep(20)

    return model, opt
