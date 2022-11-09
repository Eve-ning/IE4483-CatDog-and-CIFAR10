""" CatDogDataset implements the wrapper to load our Cat Dog Dataset as DataLoaders """

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.dataset.dataset_wrapper import DatasetWrapper
from src.utils import ROOT_DIR


@dataclass
class CatDogDataset(DatasetWrapper):
    train_path: Path = ROOT_DIR / Path("datasets") / "train"
    val_path: Path = ROOT_DIR / Path("datasets") / "val"
    classes: Tuple[str] = ('cat', 'dog')

    @property
    def train_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.ToTensor(),
                self.SquarePad(),
                transforms.Resize((self.img_size, self.img_size), InterpolationMode.NEAREST),
                *self.train_augments,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                self.SquarePad(),
                transforms.Resize((self.img_size, self.img_size), InterpolationMode.NEAREST),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    @property
    def test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.SquarePad(),
            transforms.Resize((self.img_size, self.img_size), InterpolationMode.NEAREST),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @property
    def test_ds(self):
        return self.val_ds
