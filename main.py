import os

import torch
import torchvision
from torchvision import transforms
from utils.config import settings
from utils.model import model
from utils.train import train_model


transformer = transforms.Compose(
    [
        transforms.Resize(
            (settings.SIZE_H, settings.SIZE_W)
        ),  # scaling images to fixed size
        transforms.ToTensor(),  # converting to tensors
        transforms.Normalize(
            settings.image_mean, settings.image_std
        ),  # normalize image data per-channel
    ]
)

train_dataset = torchvision.datasets.ImageFolder(
    os.path.join(settings.DATA_PATH, "train_11k"), transform=transformer
)
val_dataset = torchvision.datasets.ImageFolder(
    os.path.join(settings.DATA_PATH, "val"), transform=transformer
)

test_dataset = torchvision.datasets.ImageFolder(
    os.path.join(settings.DATA_PATH, "test_labeled"), transform=transformer
)

n_train, n_val, n_test = len(train_dataset), len(val_dataset), len(test_dataset)

train_batch_gen = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=True,
    num_workers=settings.NUM_WORKERS,
)


val_batch_gen = torch.utils.data.DataLoader(
    val_dataset, batch_size=settings.BATCH_SIZE, num_workers=settings.NUM_WORKERS
)


test_batch_gen = torch.utils.data.DataLoader(
    test_dataset, batch_size=settings.BATCH_SIZE, num_workers=settings.NUM_WORKERS
)


opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt.zero_grad()
ckpt_name = "model_base.ckpt"

model, opt = train_model(
    model, train_batch_gen, val_batch_gen, opt, ckpt_name=ckpt_name, n_epochs=15
)
