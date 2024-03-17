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
from src.helper import Model, Data, CustomCallback, Gridsearch
from src.utils import measure_time

from argparse import Namespace, ArgumentParser
import os
import warnings
import logging

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
def main(args: Namespace, params: dict) -> None:

    datamodule = Data(
        map_object=StandardMap(seed=42, params=params),
        train_size=1.0,
        plot_data=False,
        plot_data_split=False,
        print_split=False,
        params=params,
    )

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

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Autoregressor trainer",
        description="Trains an autoregression model using PyTorch Lightning",
    )

    parser.add_argument(
        "--params_dir",
        type=str,
        default="config/auto_parameters.yaml",
        help="Directory containing parameter files. (default: %(default)s)",
    )
    parser.add_argument(
        "--progress_bar",
        "-prog",
        action="store_true",
        help="Show progress bar during training. (default: False)",
    )
    parser.add_argument(
        "--accelerator",
        "-acc",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Specify the accelerator to use. Choices are 'auto', 'cpu', or 'gpu'. (default: %(default)s)",
    )
    parser.add_argument(
        "--num_devices",
        default="auto",
        help="Number of devices to use. (default: %(default)s)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "ddp_spawn"],
        help="Specify the training strategy. Choices are 'auto', 'ddp', or 'ddp_spawn'. (default: %(default)s)",
    )

    args = parser.parse_args()

    gridsearch = Gridsearch(args.params_dir, use_defaults=True)
    params: dict = gridsearch.update_params()

    main(args, params)
