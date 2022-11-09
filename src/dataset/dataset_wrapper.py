from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class DatasetWrapper(ABC):
    classes: Tuple[str]
    train_path: Path
    val_path: Path
    test_path: Path = None

    augment: bool = False

    img_size: int = 128
    batch_size: int = 64
    train_workers: int = 4
    val_workers: int = 2

    class SquarePad:
        def __call__(self, image):
            _, h, w = image.shape
            max_wh = max(w, h)
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = [hp, vp, hp, vp]
            return transforms.Pad(padding)(image)

    @property
    @abstractmethod
    def train_transform(self):
        ...

    @property
    @abstractmethod
    def test_transform(self):
        ...

    @property
    def train_augments(self):
        return [
            transforms.RandomRotation(15),
            transforms.RandomPerspective(0.2),
            transforms.GaussianBlur(5, (0.1, 0.5)),
            transforms.ColorJitter(0.2, 0.2, 0.5, 0.1),
            transforms.RandomHorizontalFlip(),
        ]

    @property
    def train_ds(self):
        return ImageFolder(self.train_path.as_posix(), self.train_transform)

    @property
    def val_ds(self):
        return ImageFolder(self.val_path.as_posix(), self.test_transform)

    @property
    def test_ds(self):
        return ImageFolder(self.test_path.as_posix(), self.test_transform)

    @property
    def train_dl(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers)

    @property
    def val_dl(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.val_workers)

    @property
    def val_shuffle_dl(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.val_workers)

    @property
    def test_dl(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.val_workers)
