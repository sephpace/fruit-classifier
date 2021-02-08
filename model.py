"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

from torch import nn
from torch.nn import functional as F


class FruitClassifier(nn.Module):
    """
    Takes in an image tensor of a fruit, outputs a classification label.

    That's it.

    N = BATCH_SIZE
    C = CLASS_COUNT

    Attributes:
        main (Sequential): The main sequential model with convolutions,
                           pooling, and ReLU.
        out (Sequential):  The output layers.
    """

    def __init__(self, class_count, drop_p=0.1):
        """
        Constructor.

        Parameters:
            class_count (int): The amount of classes.
            drop_p (float):    The dropout probability.
        """
        super().__init__()
        self.class_count = class_count

        # Set up convolutional layers
        convolutional_layers = (
            nn.Conv2d(3, 6, 5),
            nn.Conv2d(6, 9, 5),
            nn.Conv2d(9, 12, 5),
        )

        # Combine all modules into sequential
        modules = []
        for conv in convolutional_layers:
            modules.append(conv)
            modules.append(nn.MaxPool2d(2))
            modules.append(nn.ReLU())
        self.main = nn.Sequential(*modules)

        # Set up dropout for conv layers
        self.conv_drop = nn.Dropout2d(drop_p)

        # Set up output layers
        linear_layers = (
            nn.Linear(972, 486),
            nn.Linear(486, 243),
            nn.Linear(243, self.class_count),
        )
        modules = []
        for lin in linear_layers:
            modules.append(lin)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(drop_p))
        self.out = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x (Tensor): The input image tensor of size (N, 3, 100, 100).

        Returns:
            (Tensor): A softmax output of size (N, C).
        """
        y = self.main(x)
        y = self.conv_drop(y)
        y = y.flatten(start_dim=1)
        y = self.out(y)
        return F.log_softmax(y, dim=1)
