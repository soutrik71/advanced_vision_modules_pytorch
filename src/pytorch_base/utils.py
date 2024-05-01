import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    if isinstance(img, torch.Tensor):
        npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def gardcam_imageprep(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(
        img, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    )

    return img, input_tensor


def grad_cam_trainer(model, target_layers, image, input_tensor, prediction_label):
    # model targets
    targets = [ClassifierOutputTarget(prediction_label)]
    # grad cam
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(image, grayscale_cams[0, :], use_rgb=True)

    #  image display
    cam = np.uint8(255 * grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    cam_images = np.hstack((np.uint8(255 * image), cam, cam_image))

    return cam_images
