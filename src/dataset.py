from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


@dataclass
class CatDogDataset:
    dataset_path: Path = Path(__file__).parents[1] / Path("datasets")
    img_size: int = 128
    batch_size: int = 64
    train_workers: int = 4
    val_workers: int = 2

    def __post_init__(self):
        """ Post calculates some parameters.

        Notes:
            Test cannot be converted into a ImageFolder as it's not within a sub folder.
        """

        train_path: Path = self.dataset_path / "train"
        self.test_path: Path = self.dataset_path / "test"
        val_path: Path = self.dataset_path / "val"

        self.train_ds = ImageFolder(
            train_path.as_posix(),
            transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(15),
                transforms.GaussianBlur(5, (0.1, 0.5)),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.01),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ]))

        self.val_ds = ImageFolder(
            val_path.as_posix(),
            transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ])
        )

    @property
    def train_dl(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers)

    @property
    def val_dl(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.val_workers)

    @property
    def val_dl_shuffle(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.val_workers)
