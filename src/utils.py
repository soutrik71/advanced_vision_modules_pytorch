import matplotlib.pyplot as plt
import numpy as np
import torch


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    if isinstance(img, torch.Tensor):
        npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
