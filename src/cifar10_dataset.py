from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.utils import ROOT_DIR


@dataclass
class CIFAR10Dataset:
    dataset_path: Path = ROOT_DIR / "cifar10_dataset"
    img_size: int = 128
    batch_size: int = 64
    train_workers: int = 4
    test_workers: int = 2
    classes: Tuple[str] = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __post_init__(self):
        """ Post calculates some parameters.

        Notes:
            We don't have
        """
        self.train_ds = CIFAR10(root='./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(self.img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.RandomAffine(15),
                                    transforms.GaussianBlur(5, (0.1, 0.5)),
                                    transforms.ColorJitter(0.2, 0.2, 0.1, 0.01),
                                    transforms.CenterCrop(self.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
        self.test_ds = CIFAR10(root='./data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(self.img_size),
                                   transforms.CenterCrop(self.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))

    @property
    def train_dl(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers)

    @property
    def test_dl(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.test_workers)

    @property
    def test_dl_shuffle(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.test_workers)
