# import os
# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Work")

import pytorch_lightning as pl
import os

from utils.mapping_helper import StandardMap
from utils.helper import Model, Data, Gridsearch, CustomCallback

import warnings

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)
warnings.filterwarnings(
    "ignore",
    ".*The number of training batches*",
)
warnings.filterwarnings(
    "ignore",
    ".*across ranks is zero. Please make sure this was your intention*",
)

import logging

logging.getLogger("pytorch_lightning").setLevel(0)

ROOT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(ROOT_DIR, "config", "test_parameters.yaml")


if __name__ == "__main__":
    # necessary to continue training from checkpoint, else set to None
    version: str = None
    num_vertices: int = 1

    gridsearch = Gridsearch(CONFIG_DIR, num_vertices)

    for _ in range(num_vertices):
        params: dict = gridsearch.update_params()
        name: str = gridsearch.name

        map = StandardMap(seed=42, params=params)

        datamodule = Data(
            map_object=map,
            train_size=1.0,
            plot_data=False,
            plot_data_split=False,
            print_split=False,
            params=params,
        )

        model = Model(**params)

        logs_path: str = ""

        # **************** pl.callbacks ****************

        tb_logger = pl.loggers.TensorBoardLogger(
            logs_path, name=name, default_hp_metric=False
        )

        save_path: str = os.path.join(
            logs_path, name, "version_" + str(tb_logger.version)
        )

        print(f"Running version_{tb_logger.version}")
        print()

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="loss/train",  # careful
            mode="min",
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
            save_top_k=1,
        )

        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="loss/train",
            mode="min",
            min_delta=1e-8,
            patience=500,
            verbose=False,
        )

        gradient_avg_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-3)

        progress_bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=10)

        # **************** trainer ****************

        trainer = pl.Trainer(
            max_epochs=params.get("epochs"),
            precision=params.get("precision"),
            enable_progress_bar=False,
            logger=tb_logger,
            callbacks=[
                checkpoint_callback,
                early_stopping_callback,
                # progress_bar_callback,
                # gradient_avg_callback,
                CustomCallback(print=False),
            ],
        )

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
