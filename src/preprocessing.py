# transform for the dataset using Compose from albumentations
import albumentations as alb
import cv2
from albumentations.pytorch import ToTensorV2


def data_augmentations():
    """Data Augmentations for the CIFAR10 dataset"""
    train_transforms = alb.Compose(
        [
            alb.Resize(
                height=36, width=36, always_apply=True, interpolation=cv2.INTER_NEAREST
            ),
            alb.RandomCrop(height=32, width=32, always_apply=True),
            alb.Flip(p=0.5),
            alb.CoarseDropout(
                max_holes=1,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=[0.4914, 0.4822, 0.4465],
                mask_fill_value=None,
            ),
            alb.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
        ]
    )

    test_transforms = alb.Compose(
        [
            alb.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ToTensorV2(),
        ]
    )
    return train_transforms, test_transforms
