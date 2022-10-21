import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping

from src.cnn import CNN
from src.cat_dog_dataset import CatDogDataset
from src.lit_wrapper import LitWrapper
from src.plot import plot_predictions

if __name__ == '__main__':
    ds = CatDogDataset(batch_size=256)
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=4, verbose=False, mode="max")
    model = LitWrapper(CNN())
    trainer = pl.Trainer(
        max_epochs=40,
        accelerator='gpu',
        callbacks=[early_stop_callback],
        default_root_dir="cnn_logs",
    )

    trainer.fit(model, train_dataloaders=ds.train_dl, val_dataloaders=ds.val_dl)
    plot_predictions(trainer, model, class_mapping=ds.classes, val_dl=ds.val_dl_shuffle)

    plt.savefig("CNN_pred.png")


# V1 BS 256