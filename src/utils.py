from typing import List
import numpy as np
import yaml
from argparse import Namespace, ArgumentParser
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import logging


def read_yaml(parameters_path: str) -> dict:
    with open(parameters_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(file: dict, param_file_path: str) -> dict[str | float | int]:
    with open(param_file_path, "w") as f:
        yaml.dump(file, f, default_flow_style=None, default_style=None)


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


def setup_logger(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger("rnn_autoregressor")
    logger.setLevel(logging.INFO)

    try:
        os.makedirs(log_file_path)
    except FileExistsError:
        pass

    log_file_name = os.path.join(log_file_path, "logs.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


class Parameter:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type

        if self.type in ["float", "int"]:
            self.min = float("inf")
            self.max = -float("inf")

        elif self.type == "choice":
            self.value_counts = {}
            self.count = 0


def read_events_file(events_file_path: str) -> EventAccumulator:
    event_acc = EventAccumulator(events_file_path)
    event_acc.Reload()
    return event_acc


def extract_best_loss_from_event_file(events_file_path: str) -> str | float | int:
    event_values = read_events_file(events_file_path)
    for tag in event_values.Tags()["scalars"]:
        if tag == "metrics/min_train_loss":
            return {"best_loss": event_values.Scalars(tag)[-1].value}


class Gridsearch:
    def __init__(self, params_path: str, use_defaults: bool = False) -> None:
        self.path = params_path
        self.use_defaults = use_defaults

    def update_params(self) -> dict:
        params = read_yaml(self.path)
        if not self.use_defaults:
            params = self._update_params(params)

        try:
            del params["gridsearch"]
        except KeyError:
            pass

        return params

    def _update_params(self, params) -> dict:
        # don't use any seed
        rng: np.random.Generator = np.random.default_rng(None)

        for key, space in params.get("gridsearch").items():
            type = space.get("type")
            if type == "int":
                params[key] = int(rng.integers(space["lower"], space["upper"] + 1))
            elif type == "choice":
                list = space.get("list")
                choice = rng.choice(list)
                try:
                    choice = float(choice)
                except:
                    choice = str(choice)
                params[key] = choice
            elif type == "float":
                params[key] = rng.uniform(space["lower"], space["upper"])

        return params


def import_parsed_args(script_name: str) -> Namespace:
    parser = ArgumentParser(prog=script_name)

    if script_name == "Autoregressor trainer":
        parser.add_argument(
            "--num_epochs",
            type=int,
            default=1000,
            help="Number of epochs to train the model for. (default: %(default)s)",
        )
        parser.add_argument(
            "--train_size",
            type=float,
            default=0.8,
            help="Fraction of data to use for training. (default: %(default)s)",
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
            "--devices",
            default="auto",
            # nargs="*",
            type=int,
            help="Number or list of devices to use. (default: %(default)s)",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            default="auto",
            choices=["auto", "ddp", "ddp_spawn"],
            help="Specify the training strategy. Choices are 'auto', 'ddp', or 'ddp_spawn'. (default: %(default)s)",
        )
        parser.add_argument(
            "--num_nodes",
            type=int,
            default=1,
            help="Specify number of nodes to use. (default: 1)",
        )

    if script_name == "Parameter updater":
        parser.add_argument(
            "--max_good_loss",
            type=float,
            default=5e-6,
            help="Maximum loss value considered acceptable for selecting parameters. (default: %(default)s)",
        )
        parser.add_argument(
            "--min_good_samples",
            type=int,
            default=3,
            help="Minimum number of good samples required for parameter selection, otherwise parameters aren't updated, but training continues. (default: %(default)s)",
        )
        parser.add_argument(
            "--check_every_n_steps",
            type=int,
            default=1,
            help="Check for new good samples every n steps. (default: %(default)s)",
        )
        parser.add_argument(
            "--current_step",
            type=int,
            default=1,
            help="Current step of the training. (default: None)",
        )

    return parser.parse_args()
