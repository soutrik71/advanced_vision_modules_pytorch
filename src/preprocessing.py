# transform for the dataset using Compose from albumentations
import albumentations as alb
from albumentations.pytorch import ToTensorV2

mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)


def data_augmentations():
    """Data Augmentations for the CIFAR10 dataset"""
    train_transforms = alb.Compose(
        [
            alb.Normalize(mean=mean, std=std),
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
                fill_value=mean,
                mask_fill_value=None,
            ),
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
