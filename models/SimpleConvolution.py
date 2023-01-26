import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


class SimpleConvolution(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = 3,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 1,
        channels: Optional[list] = [],
        global_average_pooling: Optional[bool] = True,
        dropout: Optional[float] = 0.0,
        batch_norm: Optional[bool] = True,
        seed: int = None,
    ):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        super(SimpleConvolution, self).__init__()
        if len(channels):
            self.channels = channels
        else:
            self.channels = [64, 256, 1024, 4096, 131072]

        self.global_average_pooling = global_average_pooling
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.channels[0],
            kernel_size=kernel_size,
            padding="same",
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=kernel_size,
            padding="same",
            stride=stride,
        )
        # Until here we beat Arora with sample=8
        self.conv3 = nn.Conv2d(
            in_channels=self.channels[1],
            out_channels=self.channels[2],
            kernel_size=kernel_size,
            padding="same",
            stride=stride,
        )

        self.conv4 = nn.Conv2d(
            in_channels=self.channels[2],
            out_channels=self.channels[3],
            padding="same",
            kernel_size=kernel_size,
            stride=stride,
        )

        if len(self.channels) > 4:
            self.conv5 = nn.Conv2d(
                in_channels=self.channels[3],
                out_channels=self.channels[4],
                padding="same",
                kernel_size=kernel_size,
                stride=stride,
            )

        # It's always bad to use dropout
        self.dropout = dropout
        if dropout > 0:
            self.drop = nn.Dropout2d(self.dropout)
        self.logger = logging.getLogger("SimpleConvolution")
        self.logger.info(f"Will use Dropout: {self.dropout}\tchannels:{self.channels}")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_normalize = batch_norm

    def forward(self, x):
        # 16 x 16 x 64
        x = self.pool(F.relu(self.conv1(x)))

        if self.batch_normalize:

            batch_norm = nn.BatchNorm2d(num_features=self.channels[0])
            x = batch_norm(x)

        # 8 x 8 x 256
        x = self.pool(F.relu(self.conv2(x)))
        if self.batch_normalize:
            batch_norm = nn.BatchNorm2d(num_features=self.channels[1])
            x = batch_norm(x)

        # 4 x 4 x 1024
        x = self.pool(F.relu(self.conv3(x)))

        if self.batch_normalize:
            batch_norm = nn.BatchNorm2d(num_features=self.channels[2])
            x = batch_norm(x)
        # 2 x 2 x 16384
        x = self.pool(F.relu(self.conv4(x)))
        if self.batch_normalize:
            batch_norm = nn.BatchNorm2d(num_features=self.channels[3])
            x = batch_norm(x)

        if len(self.channels) > 4:
            x = self.pool(F.relu(self.conv5(x)))
            if self.dropout > 0:
                x = self.drop(x)
            if self.batch_normalize:
                batch_norm = nn.BatchNorm2d(num_features=self.channels[4])
                x = batch_norm(x)

        # When using [64, 256, 1024, 4096, 16384]
        # x should be 50k, 16384, 1, 1
        if self.global_average_pooling:
            size_ = x.shape[-1]
            gap = nn.AvgPool2d(kernel_size=(size_, size_))
            x = gap(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    simple_con = SimpleConvolution()
    from data.DatasetManager import DatasetsManager

    dm = DatasetsManager()

    trainloader, testloader = dm.torch_load_cifar_10_as_tensors()

    for data in trainloader:
        X_train, y_train = data

        convolved = simple_con(X_train)
