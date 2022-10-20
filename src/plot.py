from typing import Tuple

import torch
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

mapping = {0: 'cat', 1: 'dog'}


def plot_predictions(
        preds,
        val_dl: DataLoader,
        seed: int = 373,
        rows_cols: Tuple[int, int] = (8, 8),
        figsize: Tuple[int, int] = (10, 10)):
    """ Plots the predictions of a batch

    Args:
        preds: Predictions from LightningModule.predict(...)
        val_dl: Validation DataLoader (can be shuffled)
        seed: Seed of Image Shuffle
        rows_cols: Number of rows and columns
        figsize: Size of figure

    """
    seed_everything(seed)
    fig, axs = plt.subplots(*rows_cols, figsize=figsize)
    for pred_b, val_b in zip(preds, val_dl):
        for pred, val, val_ix, ax in zip(pred_b, val_b[0], val_b[1], axs.flatten()):
            val = val.swapaxes(0, -1).swapaxes(0, 1)
            pred_ix = torch.argmax(pred)
            ax.axis('off')
            ax.imshow(val)
            ax.set_title(f"{mapping[pred_ix.item()]} / {mapping[val_ix.item()]}")
        break
    _ = plt.suptitle("Predict / Actual")
    _ = plt.tight_layout()
