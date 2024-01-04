from dotenv import load_dotenv
from pydantic import BaseSettings
from torch import cuda, device


device = device("cuda") if cuda.is_available() else device("cpu")


class Settings(BaseSettings):
    DATA_PATH: str = r"data"  # PATH TO THE DATASET

    # Number of threads for data loader
    NUM_WORKERS: int = 4

    # Image size:
    SIZE_H: int = 96
    SIZE_W: int = 96

    # Number of classes in the dataset
    NUM_CLASSES: int = 2

    # Epochs:
    EPOCH_NUM: int = 30

    # Batch size:
    BATCH_SIZE: int = 256

    # Images mean and std channelwise
    image_mean: list[float] = [0.485, 0.456, 0.406]
    image_std: list[float] = [0.229, 0.224, 0.225]

    # Last layer (embeddings) size for CNN models
    EMBEDDING_SIZE: int = 128


load_dotenv()
settings = Settings()
