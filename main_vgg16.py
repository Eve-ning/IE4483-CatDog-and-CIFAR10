import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.dataset import CatDogDataset
from src.lit_wrapper import LitWrapper
from src.plot import plot_predictions
from src.vgg16 import VGG16

if __name__ == '__main__':
    ds = CatDogDataset(train_workers=1, val_workers=1, img_size=224)
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=4, verbose=False, mode="max")
    model = LitWrapper(VGG16())
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator='gpu',
        callbacks=[early_stop_callback],
        fast_dev_run=True
    )

    trainer.fit(model, train_dataloaders=ds.train_dl, val_dataloaders=ds.val_dl)
    preds = trainer.predict(model, dataloaders=ds.val_dl)
    plot_predictions(preds, val_dl=ds.val_dl_shuffle)

    plt.savefig("CNN_pred.png")


#%%
#64x1000 and 25088x4096)