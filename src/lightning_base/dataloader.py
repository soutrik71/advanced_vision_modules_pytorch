import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from src.lightning_base.augmentation import AlbumentationsCifar10, data_augmentations


class Cifar10DataModule(L.LightningDataModule):
    def __init__(
        self,
        seed,
        mean,
        std,
        data_dir="./data",
        batch_size=128,
        num_workers=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms, self.test_transforms = data_augmentations(mean, std)
        self.seed = seed

    def prepare_data(self) -> None:

        AlbumentationsCifar10(root=self.data_dir, train=True, download=True)
        AlbumentationsCifar10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):

        if stage == "train" or stage == "fit":
            mnist_full = AlbumentationsCifar10(
                root=self.data_dir, train=True, transform=self.train_transforms
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [0.7, 0.3],
                generator=torch.Generator().manual_seed(self.seed),
            )

        elif stage == "test" or stage == "predict":
            self.mnist_test = AlbumentationsCifar10(
                root=self.data_dir, train=False, transform=self.test_transforms
            )

    def train_dataloader(self):

        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):

        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):

        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):

        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
