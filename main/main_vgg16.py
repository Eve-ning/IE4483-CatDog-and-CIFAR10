import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.dataset import CatDogDataset
from src.lit_wrapper import LitWrapper
from src.plot import plot_predictions
from src.vgg16 import VGG16

if __name__ == '__main__':
    ds = CatDogDataset(train_workers=6, val_workers=2, img_size=224)
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=4, verbose=False, mode="max")
    model = LitWrapper(VGG16())
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu',
        callbacks=[early_stop_callback],
        default_root_dir="vgg_logs",
    )

    trainer.fit(model, train_dataloaders=ds.train_dl, val_dataloaders=ds.val_dl)
    plot_predictions(trainer, model, val_dl=ds.val_dl_shuffle)

    plt.savefig("VGG_pred.png")
