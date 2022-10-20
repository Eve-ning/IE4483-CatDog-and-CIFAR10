import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 3, ave_pool: bool = True):
        """ Implements the common block for our custom CNN

        Args:
            c_in: Number of Channels input
            c_out: Number of Channels output
            kernel: Size of kernel (the length of each side). Must be odd
            ave_pool: Whether to average pool after the block
        """
        super(CNNBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, 1, int((kernel - 1) // 2)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        if ave_pool:
            self.net.append(nn.AvgPool2d(2, 2))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        """ Implements our custom CNN Model, inspired by deep CNNs

        Notes:
            We noted that deep CNNs usually:
            - Increase Channels in further layers
            - Use small kernel sizes (like 3)
            - Modern Deep CNNs use padding to avoid having to correct image dimension reduction after convolution
            - Use Average/Max Pool to shrink image sizes

        """
        super(CNN, self).__init__()

        self.net = nn.Sequential(
            CNNBlock(3, 16, 7, ave_pool=False),
            CNNBlock(16, 16, 5),
            CNNBlock(16, 32),
            CNNBlock(32, 32, ave_pool=False),
            CNNBlock(32, 64),
            CNNBlock(64, 64, ave_pool=False),
            CNNBlock(64, 128),
            CNNBlock(128, 128, ave_pool=False),
            CNNBlock(128, 128),
            CNNBlock(128, 128, ave_pool=False),
            CNNBlock(128, 128),
            CNNBlock(128, 128, ave_pool=False),
            CNNBlock(128, 128),
            nn.Flatten(),
            nn.Linear(128, 2),
            nn.Dropout(0.1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)
