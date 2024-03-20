from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import callbacks

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    DeviceStatsMonitor,
)

from src.mapping_helper import StandardMap
from src.helper import Model, Data, CustomCallback
from src.utils import measure_time, read_yaml, import_parsed_args

from argparse import Namespace
import os
import warnings
import logging
from time import sleep

os.environ["GLOO_SOCKET_IFNAME"] = "en0"
warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning").setLevel(0)


def get_callbacks(tb_logger: TensorBoardLogger) -> list[callbacks]:

    save_path: str = os.path.join(tb_logger.name, "version_" + str(tb_logger.version))

    return [
        ModelCheckpoint(
            monitor="loss/train",
            mode="min",
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(
            monitor="loss/train",
            mode="min",
            min_delta=1e-8,
            patience=350,
        ),
        DeviceStatsMonitor(),
        CustomCallback(print=False),
    ]


# @measure_time
def main(
    args: Namespace, params: dict, sleep_sec: int, map_object: StandardMap
) -> None:
    sleep(sleep_sec)

    datamodule = Data(map_object=map_object, train_size=0.8, params=params)

    model = Model(**params)

    tb_logger = TensorBoardLogger(
        save_dir="", name=params.get("name"), default_hp_metric=False
    )

    trainer = Trainer(
        max_epochs=params.get("epochs"),
        precision=params.get("precision"),
        enable_progress_bar=args.progress_bar,
        logger=tb_logger,
        callbacks=get_callbacks(tb_logger),
        accelerator=args.accelerator,
        devices=args.num_devices,
        strategy=args.strategy,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Autoregressor trainer")

    params = read_yaml(args.params_dir)

    map_object = StandardMap(seed=42, params=params)

    main(args, params, 0, map_object)
