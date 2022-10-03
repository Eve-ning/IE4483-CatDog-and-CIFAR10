from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

#%%
DATASET_PATH = Path("datasets")
TRAIN_PATH = DATASET_PATH / "train"
TRAIN_CAT_PATH = TRAIN_PATH / "cat"
TRAIN_DOG_PATH = TRAIN_PATH / "dog"
TEST_PATH = DATASET_PATH / "test"
VAL_PATH = DATASET_PATH / "val"
VAL_CAT_PATH = VAL_PATH / "cat"
VAL_DOG_PATH = VAL_PATH / "dog"
#%%
train_ds = ImageFolder(
    TRAIN_PATH.as_posix(),
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
train_dl = DataLoader(train_ds)
#%%

plt.imshow(np.random.randint(0, 255, [100, 100]))
plt.show()

# Create a CNN Module
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a pytorch lightning module
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CNN()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

#%%
model = LitModel()
trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(model)

