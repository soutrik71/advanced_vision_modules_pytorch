import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from src.pytorch_base.utils import imshow


def get_data_characteristics(data, datatype="train"):
    """
    Get data characteristics
    Args:
    data (torch.utils.data.dataset.Dataset): Dataset object
    datatype (str): train or test
    """
    features = next(iter(data))[0]
    print(f"The data is [{datatype}]")
    print(f"Total no of datapoints: {len(data)}")
    print(f"The shape of the data: {features.shape}")
    print(f"The mean accross all channels: {features.mean(dim=(1, 2))}")
    print(f"The stdev accross all channels: {features.std(dim=(1, 2))}")
    print(f"The min pixel value: {features.min()}")
    print(f"The max pixel value: {features.max()}")
    print("\n")


def show_rawdata(data_set, number_of_samples, classes):
    """
    Show raw data
    """

    if number_of_samples % 2 == 0:
        batch_data = []
        batch_label = []
        for count, item in enumerate(data_set):
            if not count < number_of_samples:
                break
            batch_data.append(item[0])
            batch_label.append(item[1])

        batch_data = torch.stack(batch_data, dim=0).numpy()

        fix, axs = plt.subplots(
            ncols=number_of_samples // 2, nrows=2, sharex=True, sharey=True
        )
        k, i = 0, 0
        for img, label in list(zip(batch_data, batch_label)):
            if i != 0 and i % (number_of_samples // 2) == 0:
                k += 1
                i = 0
            axs[k, i].set_title(classes[label])
            axs[k, i].imshow(
                np.transpose(img.squeeze(), (1, 2, 0)),
                interpolation="nearest",
            )
            axs[k, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            i += 1

        plt.show()
    else:
        raise Exception("The number of samples should be divisible by 2")


def show_batchdata(batch, samples):
    """
    Show data loader batches
    """
    if samples % 10 == 0:
        plt.figure(figsize=(25, 15))
        imshow(torchvision.utils.make_grid(batch[:samples], nrow=(samples // 5)))
        plt.show()
    else:
        raise Exception("The number of samples should be divisible by 5")
