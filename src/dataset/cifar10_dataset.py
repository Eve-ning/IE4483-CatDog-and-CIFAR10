""" CIFAR10Dataset implements the wrapper to for PyTorch's CIFAR10 dataset loading. Overriding default transforms from
DatasetWrapper. """


from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode

from src.dataset.dataset_wrapper import DatasetWrapper
from src.utils import ROOT_DIR


@dataclass
class CIFAR10Dataset(DatasetWrapper):
    train_path: Path = ROOT_DIR / Path("cifar10_dataset") / "train"
    val_path: Path = ROOT_DIR / Path("cifar10_dataset") / "val"
    dataset_path: Path = ROOT_DIR / "cifar10_dataset"
    classes: Tuple[str] = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __post_init__(self):
        """ Load CIFAR 10 """
        train_ds = CIFAR10(
            root=self.dataset_path.as_posix(), train=True, download=True,
            transform=self.train_transform
        )

        train_size = int(len(train_ds) * 0.7)
        val_size = int(len(train_ds) - train_size)
        self.train_ds_, self.val_ds_ = random_split(train_ds, [train_size, val_size],
                                                    generator=torch.Generator().manual_seed(373))
        self.test_ds_ = CIFAR10(
            root=self.dataset_path.as_posix(), train=False, download=True,
            transform=self.test_transform
        )

    @property
    def train_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.ToTensor(),
                *self.train_augments,
                transforms.Resize((self.img_size, self.img_size), InterpolationMode.NEAREST),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.img_size, self.img_size), InterpolationMode.NEAREST),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    @property
    def test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @property
    def train_ds(self):
        return self.train_ds_

    @property
    def val_ds(self):
        return self.val_ds_

    @property
    def test_ds(self):
        return self.test_ds_
