import time
from datetime import timedelta
from typing import Callable
import yaml
from argparse import Namespace, ArgumentParser


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


def import_parsed_args(script_name: str) -> Namespace:
    script_names = [
        "Autoregressor trainer",
        "Parameter updater",
        "Hyperparameter optimizer",
    ]
    parser = ArgumentParser(
        prog=script_name,
        description="Trains an autoregression model using PyTorch Lightning",
    )

    if script_name == script_names[0]:
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

    elif script_name == script_names[1]:
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

    elif script_name == script_names[2]:
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
