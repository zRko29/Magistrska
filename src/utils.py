import time
from datetime import timedelta
from typing import Callable, List
import yaml
from argparse import Namespace, ArgumentParser
import os

import logging


logger = logging.getLogger("rnn_autoregressor")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("logs/log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def read_yaml(parameters_path: str) -> dict:
    with open(parameters_path, "r") as file:
        return yaml.safe_load(file)


def measure_time(func: Callable) -> Callable:
    """
    A decorator that measures the time a function takes to run.
    """

    def wrapper(*args, **kwargs):
        print("\n------------------------------------")
        print('Started function "{}"!\n'.format(func.__name__))
        t1 = time.time()
        val = func(*args, **kwargs)
        t2 = time.time() - t1
        print('\nFunction "{}" finished!'.format(func.__name__))
        print(f"Function ran for: {timedelta(seconds=t2)}")
        print("------------------------------------\n")
        return val

    return wrapper


def get_inference_folders(directory_path: str, version: str) -> List[str]:
    if version is not None:
        folders: List[str] = [os.path.join(directory_path, f"version_{version}")]
    else:
        folders: List[str] = [
            os.path.join(directory_path, folder)
            for folder in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, folder))
        ]
        folders.sort()
    return folders


def import_parsed_args(script_name: str) -> Namespace:
    parser = ArgumentParser(prog=script_name)

    parser.add_argument(
        "--params_dir",
        type=str,
        default="config/parameters.yaml",
        help="Directory containing parameter files. (default: %(default)s)",
    )

    if script_name in ["Autoregressor trainer", "Hyperparameter optimizer"]:
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

    if script_name in ["Parameter updater", "Hyperparameter optimizer"]:
        parser.add_argument(
            "--max_loss",
            type=float,
            default=1e-6,
            help="Maximum loss value considered acceptable for selecting parameters. (default: %(default)s)",
        )
        parser.add_argument(
            "--min_good_samples",
            type=int,
            default=3,
            help="Minimum number of good samples required for parameter selection, otherwise parameters aren't updated, but training continues. (default: %(default)s)",
        )

    if script_name == "Hyperparameter optimizer":
        parser.add_argument(
            "--optimization_steps",
            type=int,
            default=5,
            help="Number of optimization steps to perform. (default: %(default)s)",
        )
        parser.add_argument(
            "--models_per_step",
            type=int,
            default=5,
            help="Number of models to train in each optimization step. (default: %(default)s)",
        )

    return parser.parse_args()
