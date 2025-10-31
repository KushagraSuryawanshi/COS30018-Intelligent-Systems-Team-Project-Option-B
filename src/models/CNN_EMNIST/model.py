import torch
import torch.nn as nn

"""
Convolutional Neural Network for EMNIST ByClass classification

Classes:
 - Net: Defines an improved CNN model for 62-class EMNIST dataset
"""


class Net(nn.Module):
    """
    Improved/New CNN architecture for EMNIST ByClass dataset (62 classes).

    Architecture:
     - Three convolutional blocks with BatchNorm, ReLU, MaxPool, and Dropout.
     - Global average pooling layer.
     - Fully connected block with Dropout regularization.

    :return: CNN model instance.
    """

    def __init__(self):
        """Initialize convolutional and fully connected layers."""
        super(Net, self).__init__()

        # First convolutional block: 1 → 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )

        # Second convolutional block: 32 → 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )

        # Third convolutional block: 64 → 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Global Average Pooling layer to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected block with dropout for regularization
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 62)
        )

    def forward(self, x):
        """
        Define the forward pass of the CNN.

        :param x: Input tensor of shape (batch_size, 1, 28, 28).
        :return: Output tensor of shape (batch_size, 62), logits for 62 classes.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.fc(x)
        return x