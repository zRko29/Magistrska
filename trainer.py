from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import callbacks

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch

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
            monitor=args.monitor,
            mode=args.mode,
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
        ),
        # EarlyStopping(
        #     monitor=args.monitor,
        #     mode=args.mode,
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

    if args.accelerator == "cpu":
        args.devices = "auto"

    trainer = Trainer(
        max_epochs=args.epochs,
        precision=params.get("precision"),
        logger=tb_logger,
        callbacks=get_callbacks(args, save_path),
        deterministic=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        enable_progress_bar=args.progress_bar,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
    )

    if trainer.is_global_zero:
        print(f"Running version_{tb_logger.version}.")
        logger.info(f"Running trainer.py (version_{tb_logger.version}).")

        print_args = args.__dict__.copy()
        del print_args["path"]
        logger.info(f"args = {print_args}")

    if args.static_model:
        from src.StaticModel import Model
    elif params.get("rnn_type") == "vanilla":
        from src.VanillaRNN import Model
    elif params.get("rnn_type") == "stacked":
        from src.StackedRNN import Model
    elif params.get("rnn_type") == "hybrid":
        from src.HybridRNN import Model

    if args.ckpt_path:
        model = Model.load_from_checkpoint(args.ckpt_path, map_location="cpu")
    else:
        model = Model(**params)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Autoregressor trainer")
    args.path = os.path.abspath(args.path)

    logger = setup_logger(args.path, "rnn_autoregressor")

    params_path = os.path.join(args.path, "current_params.yaml")
    params = read_yaml(params_path)

    main(args, params)
