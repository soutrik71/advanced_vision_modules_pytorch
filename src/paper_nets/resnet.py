from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResnetBasicBlock(nn.Module):
    """
    Builds the Basic Block of the ResNet model.
    For ResNet18 and ResNet34, these are stackings od 3x3=>3x3 convolutional
    layers.
    For ResNet50 and above, these are stackings of 1x1=>3x3=>1x1 (BottleNeck)
    layers.
    """

    def __init__(
        self,
        is_Bottleneck: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        """
        Initializes the ResnetBasicBlock class.
        Args:
            is_Bottleneck (int): If True, uses the BottleNeck architecture.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layers.
            expansion (int): Multiplicative factor for the subsequent conv2d
                layer's output channels.
            downsample (nn.Module): Downsample the input.
        """
        super(ResnetBasicBlock, self).__init__()
        self.is_Bottleneck = is_Bottleneck
        # Multiplicative factor for the subsequent conv2d layer's output
        # channels.
        # It is 1 for ResNet18 and ResNet34, and 4 for the others.
        self.expansion = expansion
        self.downsample = downsample
        # 1x1 convolution for ResNet50 and above.
        if is_Bottleneck:
            self.conv0 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.bn0 = nn.BatchNorm2d(out_channels)
            in_channels = out_channels
        # Common 3x3 convolution for all.
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 1x1 convolution for ResNet50 and above.
        if is_Bottleneck:
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        else:
            # 3x3 convolution for ResNet18 and ResNet34 and above.
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Through 1x1 convolution if ResNet50 or above.
        if self.is_Bottleneck:
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.relu(out)
        # Use the above output if ResNet50 and above.
        if self.is_Bottleneck:
            out = self.conv1(out)
        # Else use the input to the `forward` method.
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResnetPrepBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        """
        Prepares the input for the ResNet model.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class ResNetX(nn.Module):
    def __init__(
        self,
        resnet_variant,
        img_channels: int,
        num_classes: int = 1000,
    ) -> None:
        """
        Initializes the ResNet model.
        Args:
            resnet_variant (str): Variant of ResNet model.
            img_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
        """
        super(ResNetX, self).__init__()
        self.channels_list = resnet_variant[0]
        self.repeatition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_Bottleneck = resnet_variant[3]

        self.in_channels = 64
        # prep layer
        self.prep_layer = ResnetPrepBlock(img_channels, self.in_channels)
        # layer 1
        self.layer1 = self._make_layer(64, self.repeatition_list[0], stride=1)
        # layer 2
        self.layer2 = self._make_layer(
            128,
            self.repeatition_list[1],
            stride=2,
        )
        # layer 3
        self.layer3 = self._make_layer(
            256,
            self.repeatition_list[2],
            stride=2,
        )
        # layer 4
        self.layer4 = self._make_layer(
            512,
            self.repeatition_list[3],
            stride=2,
        )

        # classification layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * self.expansion, num_classes),
        )

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        Creates a layer of the ResNet model.
        Args:
            out_channels (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the convolutional layers.
        Returns:
            nn.Sequential: A layer of the ResNet model.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            ResnetBasicBlock(
                self.is_Bottleneck,
                self.in_channels,
                out_channels,
                stride,
                self.expansion,
                downsample,
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_blocks):
            layers.append(
                ResnetBasicBlock(
                    self.is_Bottleneck,
                    self.in_channels,
                    out_channels,
                    stride=1,
                    expansion=self.expansion,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x
