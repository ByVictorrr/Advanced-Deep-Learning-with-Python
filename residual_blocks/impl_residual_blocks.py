import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms


class PreActivationBlock(nn.Module):
    """Defines a pre-activation residual block for a non-bottleneck architecture.

    EXPANSION determines how many times the number of slices (channels) is expanded
    For a non-bottleneck block, EXPANSION is 1 (no channel expansion).
    """
    EXPANSION = 1

    def __init__(self, in_slices, slices, stride=1):
        """Initializes the block with the required layers and parameters.

        :param in_slices: Number of input channels (feature maps) to the block.
        :param slices: Number of output channels for the main convolutional layers.
        :param stride: Controls the downsampling factor for the block (default: 1, no downsampling).
        """
        super().__init__()  # Calls the constructor of nn.Module.

        # First Batch Normalization layer for input normalization.
        self.bn_1 = nn.BatchNorm2d(in_slices)

        # First 3x3 convolutional layer:
        # - in_channels: Number of input channels (in_slices).
        # - out_channels: Number of output channels (slices).
        # - kernel_size: Size of the convolution kernel (3x3).
        # - stride: Controls downsampling.
        # - padding: 1 ensures the spatial dimensions remain the same.
        # - bias: False, as bias is unnecessary with batch normalization.
        self.conv_1 = nn.Conv2d(
            in_channels=in_slices,
            out_channels=slices,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        # Second Batch Normalization layer for normalizing the output of the first convolution.
        self.bn_2 = nn.BatchNorm2d(slices)

        # Second 3x3 convolutional layer:
        # - in_channels: Number of input channels (slices).
        # - out_channels: Number of output channels (slices).
        # - kernel_size: 3x3 kernel.
        # - stride: Fixed to 1 (no additional downsampling).
        # - padding: 1 maintains spatial dimensions.
        # - bias: False, as batch normalization handles bias.
        self.conv_2 = nn.Conv2d(
            in_channels=slices,
            out_channels=slices,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Shortcut connection:
        # If the input and output dimensions are different (either due to stride or channel mismatch),
        # use a 1x1 convolution to match dimensions.
        if stride != 1 or in_slices != self.EXPANSION * slices:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_slices,  # Match the input channels.
                    out_channels=self.EXPANSION * slices,  # Match the output channels.
                    kernel_size=1,  # 1x1 convolution for dimensionality adjustment.
                    stride=stride,  # Match the downsampling factor.
                    bias=False,  # Bias is not needed with batch normalization.
                )
            )
        else:
            # If dimensions already match, the shortcut is just the identity mapping.
            self.shortcut = nn.Sequential()

    def forward(self, x):
        """Defines the forward pass of the pre-activation block.

        :param x: Input tensor of shape (N, C, H, W), where
                  N is the batch size,
                  C is the number of input channels,
                  H is the height, and
                  W is the width.
        :return: Output tensor of the same spatial dimensions as the input, but possibly with
                 adjusted channels if downsampling or dimensional changes are applied.
        """
        # Apply the first batch normalization and activation function (ReLU).
        out = F.relu(self.bn_1(x))

        # Determine the shortcut path:
        # - If a dimensional mismatch exists (e.g., due to stride or channel changes), use the
        #   shortcut defined during initialization.
        # - Otherwise, pass the input directly through as the identity mapping.
        shortcut = self.shortcut(out) if self.shortcut else x

        # Apply the first convolution (3x3) to the normalized and activated input.
        out = self.conv_1(out)

        # Apply the second batch normalization and activation function (ReLU).
        out = F.relu(self.bn_2(out))

        # Apply the second convolution (3x3).
        out = self.conv_2(out)

        # Add the shortcut connection to the output of the main path.
        out += shortcut

        # Return the final output of the block.
        return out


class PreActivationBottleneckBlock(nn.Module):
    EXPANSION = 4

    def __init__(self, in_slices, slices, stride=1):
        super().__init__()
        self.bn_1 = nn.BatchNorm2d(in_slices)
        self.conv_1 = nn.Conv2d(
            in_channels=in_slices,
            out_channels=slices,
            kernel_size=1,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(slices)
        self.conv_2 = nn.Conv2d(
            in_channels=slices,
            out_channels=slices,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(slices)
        self.conv_3 = nn.Conv2d(
            in_channels=slices,
            out_channels=self.EXPANSION * slices,
            kernel_size=1,
            bias=False,
        )
        # if the input/output dimensions differ use convolution for the shortcut
        if stride != 1 or in_slices != self.EXPANSION * slices:
            self.shortcut = nn.Sequential(nn.Conv2d(
                in_channels=in_slices,
                out_channels=self.EXPANSION * slices,
                kernel_size=1,
                stride=stride,
                bias=False,
            ))

        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn_1(x))
        # reuse bn+relu in down sampling layers
        shortcut = self.shortcut(out) if self.shortcut else x
        out = self.conv_1(out)
        out = F.relu(self.bn_2(out))
        out = self.conv_2(out)

        out = F.relu(self.bn_3(out))
        out = self.conv_3(out)
        out += shortcut
        return out


class PreActivationResNet(nn.Module):

    def _make_group(self, block, slices, num_blocks, stride):
        """"Create one residual group."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_slices, slices, stride))
            self.in_slices = slices * block.EXPANSION
        return nn.Sequential(*layers)

    def __init__(self, block, num_blocks, num_classes=10):
        """
        :param block: type of residual block (regular or bottleneck)
        :param num_blocks: a list with 4 integer values. Each value reflects the number of residual blocks in the group
        :param num_classes:  number of output classes
        """
        super().__init__()
        self.in_slices = 64
        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.layer_1 = self._make_group(block, 64, num_blocks[0], stride=1)
        self.layer_2 = self._make_group(block, 128, num_blocks[1], stride=2)
        self.layer_3 = self._make_group(block, 256, num_blocks[2], stride=2)
        self.layer_4 = self._make_group(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.EXPANSION, num_classes)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def pre_activation_resnet_18():
    return PreActivationResNet(block=PreActivationBlock,
                               num_blocks=[2, 2, 2, 2])


def pre_activation_resnet_34():
    return PreActivationResNet(block=PreActivationBlock,
                               num_blocks=[3, 4, 6, 3])


def pre_activation_resnet_50():
    return PreActivationResNet(block=PreActivationBottleneckBlock,
                               num_blocks=[3, 4, 6, 3])


def pre_activation_resnet_101():
    return PreActivationResNet(block=PreActivationBottleneckBlock,
                               num_blocks=[3, 4, 23, 3])


def pre_activation_resnet_152():
    return PreActivationResNet(block=PreActivationBottleneckBlock,
                               num_blocks=[3, 8, 36, 3])


def train_model(model, loss_function, optimizer, data_loader):
    """Train one epoch."""
    # set model in training mode
    model.train()


    pass
def test_model(model, loss_function, data_loader):
    pass
def plot_accuracy(accuracy: list):
    """Plot accuracy"""
    plt.figure()
    plt.plot(accuracy)
    plt.xticks(
        [i for i in range(0, len(accuracy))],
        [i + 1 for i in range(0, len(accuracy))])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
if __name__ == "__main__":
    pass
