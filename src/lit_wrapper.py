import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LitWrapper(pl.LightningModule):
    def __init__(self, model, seed: int = 373):
        """ This wraps respective PyTorch models into a LightningModule, which creates commonly used functions.

        Args:
            model: The PyTorch model class INSTANCE
            seed: Seed to train the model in
        """
        super().__init__()
        seed_everything(seed)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y_true)
        self.log("train_acc", self.acc(y_pred, y_true))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        self.log("val_acc", self.acc(y_pred, y_true))

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=0.001)

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=2, verbose=True),
                "monitor": "val_acc",
            },
        }

    @staticmethod
    def acc(y_pred: torch.Tensor, y_true: torch.Tensor):
        y_pred_ix = torch.argmax(y_pred, dim=1)
        return torch.sum(y_pred_ix == y_true) / y_true.shape[0]
