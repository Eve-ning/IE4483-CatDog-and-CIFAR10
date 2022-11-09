from typing import Tuple

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader

from src.model.lit_wrapper import LitWrapper


def plot_predictions(
        trainer: pl.Trainer,
        model: LitWrapper,
        true_dl: DataLoader,
        class_mapping: Tuple[str],
        seed: int = 373,
        rows_cols: Tuple[int, int] = (8, 8),
        figsize: Tuple[int, int] = (10, 12)
):
    """ Plots the predictions of a batch in a grid

    Args:
        trainer: PyTorch Lightning Trainer
        model: PyTorch Lightning Module Instance
        true_dl: DataLoader (can be shuffled)
        class_mapping: Tuple of classes mapped to indices
        seed: Seed of Image Shuffle
        rows_cols: Number of rows and columns
        figsize: Size of figure

    """
    fig, axs = plt.subplots(*rows_cols, figsize=figsize)
    seed_everything(seed)
    preds = trainer.predict(model, true_dl)
    seed_everything(seed)
    for pred_b, true_b in zip(preds, true_dl):
        for pred, true, true_ix, ax in zip(pred_b, true_b[0], true_b[1], axs.flatten()):
            true = true.swapaxes(0, -1).swapaxes(0, 1)
            pred_ix = torch.argmax(pred)
            ax.axis('off')
            ax.imshow(minmax_scale(true.reshape(-1, 3)).reshape(true.shape))
            ax.set_title(f"{class_mapping[pred_ix.item()]} / {class_mapping[true_ix.item()]}")
        break
    _ = plt.suptitle("Predict / Actual")
    _ = plt.tight_layout()
    _ = plt.subplots_adjust(top=0.95)
