from dataclasses import dataclass


@dataclass
class Config:
    data_path: str
    num_workers: int
    size_h: int
    size_w: int
    epoch_num: int
    batch_size: int
    image_mean: list[float]
    image_std: list[float]
    embedding_size: int
    model_ckpt: str
    infer_labels: str
