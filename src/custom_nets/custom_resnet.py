import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """initializes the resnet block with a given number of channels and a given number of output channels
        Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        """
        super(ResnetBlock, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.resnet_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """forward pass of the resnet block
        Args:
        x (torch.Tensor): input tensor
        """
        interm_op = self.conv_layer1(x)
        residual_op = self.resnet_block1(x)
        # print(interm_op.shape)
        # print(residual_op.shape)
        op = residual_op + interm_op
        return op


class ResnetNetwork(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        """initializes the resnet network with a given number of input channels and a given number of classes
        Args:

        in_channels: number of input channels
        num_classes (int): number of classes
        """
        super(ResnetNetwork, self).__init__()
        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.presentation_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.resnet_layer1 = ResnetBlock(in_channels=64, out_channels=128)

        self.norm_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.resnet_layer2 = ResnetBlock(in_channels=256, out_channels=512)

        self.pooling = nn.MaxPool2d(kernel_size=(4, 4))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """forward pass of the resnet network
        Args:
        x (torch.Tensor): input tensor
        """
        op = self.presentation_layer(x)
        op = self.resnet_layer1(op)
        op = self.norm_conv1(op)
        op = self.resnet_layer2(op)
        op = self.pooling(op)
        op = self.classifier(op)
        return F.log_softmax(op, dim=-1)
