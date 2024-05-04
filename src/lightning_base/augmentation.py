import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision import datasets


def data_augmentations(mean, std):
    """Data Augmentations for the CIFAR10 dataset"""
    train_transforms = alb.Compose(
        [
            alb.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
            alb.RandomCrop(height=32, width=32, always_apply=True),
            alb.HorizontalFlip(),
            alb.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=tuple(x * 255 for x in mean),
                mask_fill_value=None,
            ),
            alb.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    test_transforms = alb.Compose(
        [
            alb.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return train_transforms, test_transforms


class AlbumentationsCifar10(datasets.CIFAR10):
    """
    AlbumentationsCifar10 dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        super(AlbumentationsCifar10, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, target
