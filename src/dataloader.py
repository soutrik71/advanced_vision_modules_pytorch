import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class CifarDS(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def get_data_loader(
    train_dateset: datasets,
    test_dateset: datasets,
    train_transforms: list,
    test_transforms: list,
    dataloader_args: dict,
):
    """Data loader for the CIFAR10 dataset"""
    # Loading custom datasets
    train_dateset = CifarDS(train_dateset.data, train_dateset.targets, train_transforms)
    test_dateset = CifarDS(test_dateset.data, test_dateset.targets, test_transforms)
    # train dataloader
    train_loader = DataLoader(train_dateset, **dataloader_args)
    # test dataloader
    test_loader = DataLoader(test_dateset, **dataloader_args)

    return train_loader, test_loader
