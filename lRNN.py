# import os
# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Work")

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
import time, os, yaml

from utils.mapping_helper import StandardMap
from utils.ltraining_helper import lModel, Data, return_elapsed_time

ROOT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

with open(os.path.join(CONFIG_DIR, "parameters.yaml"), "r") as file:
    params = yaml.safe_load(file)

if __name__ == "__main__":
    map = StandardMap(seed=42)
    data = Data(map, plot_data=False, **params)
    data.prepare_data(shuffle=True, t=0.8, input_size=1.0, plot_data_split=True)

    model = lModel(**params)

    logs_path = "logs"
    version = str(int(time.time()))
    name = "masters"
    save_path = os.path.join(logs_path, name, version)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="lmodel",
        save_on_train_epoch_end=True,
        save_top_k=1,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )

    early_stopping_callback = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=10,
        verbose=False,
        mode="min",
    )

    tb_logger = TensorBoardLogger(
        logs_path, version=version, name=name, default_hp_metric=False
    )

    timer_callback = callbacks.Timer()

    progress_bar_callback = callbacks.TQDMProgressBar(refresh_rate=50)

    trainer = pl.Trainer(
        max_epochs=params.get("machine_learning_parameters").get("epochs"),
        enable_progress_bar=True,
        logger=tb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            progress_bar_callback,
            timer_callback,
        ],
    )

    trainer.fit(model, data.train_loader, data.val_loader)
    return_elapsed_time(timer_callback.time_elapsed())
