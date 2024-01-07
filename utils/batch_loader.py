import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class BatchLoader:
    def __init__(
        self,
        image_size_h: int,
        image_size_w: int,
        image_mean: list[float],
        image_std: list[float],
        data_path: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.transformer = transforms.Compose(
            [
                transforms.Resize(
                    (image_size_h, image_size_w)
                ),  # scaling images to fixed size
                transforms.ToTensor(),  # converting to tensors
                transforms.Normalize(
                    image_mean, image_std
                ),  # normalize image data per-channel
            ]
        )

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_batch_gen(self) -> DataLoader:
        train_dataset = ImageFolder(
            os.path.join(self.data_path, "train_11k"), transform=self.transformer
        )
        train_batch_gen = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_batch_gen

    def get_test_batch_gen(self) -> DataLoader:
        test_dataset = ImageFolder(
            os.path.join(self.data_path, "test_labeled"), transform=self.transformer
        )
        test_batch_gen = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return test_batch_gen

    def get_val_batch_gen(self) -> DataLoader:
        val_dataset = ImageFolder(
            os.path.join(self.data_path, "val"), transform=self.transformer
        )
        val_batch_gen = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_batch_gen
