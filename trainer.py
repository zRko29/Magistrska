from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import callbacks

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    DeviceStatsMonitor,
)

from src.mapping_helper import StandardMap
from src.data_helper import Data
from src.utils import import_parsed_args, read_yaml, setup_logger

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
seed_everything(42, workers=True)


def get_callbacks(args: Namespace, save_path: str) -> List[callbacks]:
    return [
        ModelCheckpoint(
            monitor=args.monitor_checkpoint,
            mode=args.mode_checkpoint,
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
        ),
        DeviceStatsMonitor(),
        # EarlyStopping(
        #     monitor=args.monitor_stopping,
        #     mode=args.mode_stopping,
        #     min_delta=1e-8,
        #     patience=1000,
        # ),
    ]


def main(args: Namespace, params: dict) -> None:

    logger = logging.getLogger("rnn_autoregressor")

    map_object = StandardMap(seed=42, params=params)

    datamodule = Data(
        map_object=map_object,
        train_size=args.train_size,
        params=params,
        plot_data=False,
    )

    tb_logger = TensorBoardLogger(save_dir="", name=args.path, default_hp_metric=False)

    save_path: str = os.path.join(tb_logger.name, f"version_{tb_logger.version}")

    trainer = Trainer(
        max_epochs=args.epochs,
        precision=params.get("precision"),
        logger=tb_logger,
        callbacks=get_callbacks(args, save_path),
        deterministic=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        enable_progress_bar=args.progress_bar,
        devices=args.devices,
        num_nodes=args.num_nodes,
        profiler="simple",
    )

    if trainer.is_global_zero:
        print(f"Running version_{tb_logger.version}.")
        logger.info(f"Running trainer.py (version_{tb_logger.version}).")

        print_args = args.__dict__.copy()
        del print_args["path"]
        logger.info(f"args = {print_args}")

    model = get_model(args, params)

    trainer.fit(model, datamodule)


def get_model(args: Namespace, params: Dict) -> None:
    if args.static_model:
        from src.StaticModel import Model
    elif args.static_model2:
        from src.StaticModel2 import Model
    elif params.get("rnn_type") == "vanilla":
        from src.VanillaRNN import Model
    elif params.get("rnn_type") == "hybrid":
        from src.HybridRNN import Model
    elif params.get("rnn_type") == "MGU":
        from src.MGU import Model

    if args.checkpoint_path:
        model = Model.load_from_checkpoint(args.checkpoint_path, map_location="cpu")
    else:
        model = Model(**params)

    return model


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Autoregressor trainer")
    args.path = os.path.abspath(args.path)

    logger = setup_logger(args.path, "rnn_autoregressor")

    params_path = os.path.join(args.path, "parameters.yaml")
    params = read_yaml(params_path)

    main(args, params)
