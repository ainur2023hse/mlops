import torch
import torch.nn as nn
import torch.optim as optim
from torch import device


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


class Model:
    loss_function = nn.CrossEntropyLoss()

    def __init__(
        self,
        image_size_h: int,
        image_size_w: int,
        embedding_size: int,
        num_classes: int,
        _device: device,
    ) -> None:
        model = nn.Sequential()
        model.add_module("flatten", Flatten())

        # dense "head"
        model.add_module("dense1", nn.Linear(3 * image_size_h * image_size_w, 256))
        model.add_module("dense1_relu", nn.ReLU())
        model.add_module("dropout1", nn.Dropout(0.1))
        model.add_module("dense3", nn.Linear(256, embedding_size))
        model.add_module("dense3_relu", nn.ReLU())
        model.add_module("dropout3", nn.Dropout(0.1))
        # logits for NUM_CLASSES=2: cats and dogs
        model.add_module("dense4_logits", nn.Linear(embedding_size, num_classes))

        self.model = model.to(_device)
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.optimizer.zero_grad()
        self.loss: any = None

    def compute_loss(self, data_batch: dict) -> None:
        img_batch = data_batch["img"]
        label_batch = data_batch["label"]
        logits = self.model(img_batch)
        self.loss = self.loss_function(logits, label_batch)
