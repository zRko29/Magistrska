from __future__ import annotations
from typing import TYPE_CHECKING, List

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
from src.utils import (
    measure_time,
    read_yaml,
    import_parsed_args,
    setup_logger,
)

from argparse import Namespace
import os
import warnings
import logging

os.environ["GLOO_SOCKET_IFNAME"] = "en0"
warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)

logging.getLogger("pytorch_lightning").setLevel("INFO")


def get_callbacks(save_path: str) -> List[callbacks]:
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
        # DeviceStatsMonitor(),
        CustomCallback(),
    ]


@measure_time
def main(
    args: Namespace,
    params: dict,
    logger: logging.Logger,
) -> None:

    map_object = StandardMap(seed=42, params=params)

    datamodule = Data(
        map_object=map_object,
        train_size=args.train_size,
        params=params,
    )

    model = Model(**params)

    tb_logger = TensorBoardLogger(
        save_dir="", name=params.get("name"), default_hp_metric=False
    )

    save_path: str = os.path.join(tb_logger.name, "version_" + str(tb_logger.version))

    logger.info(f"Training version_{tb_logger.version}.")

    trainer = Trainer(
        max_epochs=args.num_epochs,
        precision=params.get("precision"),
        logger=tb_logger,
        check_val_every_n_epoch=10,
        callbacks=get_callbacks(save_path),
        enable_progress_bar=args.progress_bar,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Autoregressor trainer")

    params_dir = os.path.abspath("config")

    params_path = os.path.join(params_dir, "current_params.yaml")
    params = read_yaml(params_path)

    params["name"] = os.path.abspath(params["name"])

    logger = setup_logger(params["name"])
    logger.info("Running trainer.py")
    logger.info(f"{args.__dict__=}")

    run_time = main(args, params, logger)
