""" VGG16 wraps torchvision.models.vgg16 to adjust the last layer to desired dimensions.

Furthermore, it freezes the feature extraction layers."""

import torch.nn as nn
from torchvision.models import VGG16_Weights
from torchvision.models import vgg16


class VGG16(nn.Module):
    def __init__(self, out_dims: int = 2):
        """ Implements the pretrained VGG16 trained on the ImageNet 1K Dataset.

        Notes:
            This freezes all prior layers that extracts important features from the images.
            This will also add a final Linear layer to it

        Args:
            out_dims: How many categories to output. E.g. Cat & Dog will use 2.
        """
        super(VGG16, self).__init__()
        self.net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for p in self.net.parameters():
            p.requires_grad = False

        self.net.classifier = nn.Sequential(
            *[_ for _ in self.net.classifier.children()][:-1],
            nn.Linear(4096, out_dims),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
