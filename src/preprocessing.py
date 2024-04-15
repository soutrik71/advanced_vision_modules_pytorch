# transform for the dataset using Compose from albumentations
import albumentations as alb
from albumentations.pytorch import ToTensorV2


def data_augmentations():
    """Data Augmentations for the CIFAR10 dataset"""
    train_transforms = alb.Compose(
        [
            alb.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
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
                fill_value=[0.4914, 0.4822, 0.4465],
                mask_fill_value=None,
            ),
            ToTensorV2(),
        ]
    )

    test_transforms = alb.Compose(
        [
            alb.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
            ToTensorV2(),
        ]
    )
    return train_transforms, test_transforms
