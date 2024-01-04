import torch
import torch.nn as nn

from .config import device, settings


class Flatten(nn.Module):
    def forward(self, x):
        # finally we have it in pytorch
        return torch.flatten(x, start_dim=1)


model = nn.Sequential()

# reshape from "images" to flat vectors
model.add_module("flatten", Flatten())

# dense "head"
model.add_module("dense1", nn.Linear(3 * settings.SIZE_H * settings.SIZE_W, 256))
model.add_module("dense1_relu", nn.ReLU())
model.add_module("dropout1", nn.Dropout(0.1))
model.add_module("dense3", nn.Linear(256, settings.EMBEDDING_SIZE))
model.add_module("dense3_relu", nn.ReLU())
model.add_module("dropout3", nn.Dropout(0.1))
# logits for NUM_CLASSES=2: cats and dogs
model.add_module(
    "dense4_logits", nn.Linear(settings.EMBEDDING_SIZE, settings.NUM_CLASSES)
)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()


def compute_loss(model, data_batch):
    # load the data
    img_batch = data_batch["img"]
    label_batch = data_batch["label"]
    # forward pass
    logits = model(img_batch)

    # loss computation
    loss = loss_function(logits, label_batch)

    return loss, model
