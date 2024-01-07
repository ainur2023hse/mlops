import sys

import fire
import git
import hydra
import mlflow
from dvc.repo import Repo
from hydra.core.config_store import ConfigStore
from torch import cuda, device
from torch import load as t_load
from utils.batch_loader import BatchLoader
from utils.config import Config
from utils.model import Model
from utils.test import test
from utils.train import train


cs = ConfigStore.instance()
cs.store(name="mnist_config", node=Config)
config_path = "config.yaml"


class Namer(object):
    def __init__(self, cfg: Config) -> None:
        self.repo = Repo(".")
        self.repo.pull()
        self.cfg = cfg
        self.device = device("cuda") if cuda.is_available() else device("cpu")
        self.model = Model(cfg.size_h, cfg.size_w, cfg.embedding_size, 2, self.device)

        self.batch_loader = BatchLoader(
            image_size_h=cfg.size_h,
            image_size_w=cfg.size_w,
            image_std=cfg.image_std,
            image_mean=cfg.image_mean,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            data_path=cfg.data_path,
        )

    def train(self) -> None:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        with mlflow.start_run():
            mlflow.log_param("git hash", sha)
            mlflow.log_artifact(config_path)
            train(
                model=self.model,
                _device=self.device,
                train_batch_generator=self.batch_loader.get_train_batch_gen(),
                val_batch_generator=self.batch_loader.get_val_batch_gen(),
                ckpt_name=self.cfg.model_ckpt,
                n_epochs=2,
            )
            self.repo.add(self.cfg.model_ckpt)
            self.repo.push()

    def infer(self) -> None:
        with open(self.cfg.model_ckpt, "rb") as f:
            self.model.model = t_load(f)
        labels = test(
            model=self.model,
            batch_generator=self.batch_loader.get_test_batch_gen(),
            _device=self.device,
        )["labels"]
        with open(self.cfg.infer_labels, "w") as f:
            f.write("\n".join((str(label) for label in labels)))
        self.repo.add(self.cfg.infer_labels)
        self.repo.push()


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    sys.argv = sys_argv[0:2]
    namer = Namer(cfg)
    fire.Fire(namer)


if __name__ == "__main__":
    sys_argv = sys.argv
    sys.argv = [sys_argv[0], *sys_argv[2:]]
    main()
