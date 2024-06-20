from typing import List
import numpy as np
import yaml
from argparse import Namespace, ArgumentParser
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import matplotlib.pyplot as plt
import functools
import logging


def read_yaml(parameters_path: str) -> dict:
    with open(parameters_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(file: dict, param_file_path: str) -> dict[str | float | int]:
    with open(param_file_path, "w") as f:
        yaml.dump(file, f, default_flow_style=None, default_style=None, sort_keys=False)


def get_inference_folders(directory_path: str, version: str) -> List[str]:
    if version is not None:
        folders: List[str] = [os.path.join(directory_path, f"version_{version}")]
    else:
        folders: List[str] = [
            os.path.join(directory_path, folder)
            for folder in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, folder))
        ]
        folders = [int(i.split("_")[-1]) for i in folders]
        folders.sort()
        folders = [os.path.join(directory_path, f"version_{i}") for i in folders]
    return folders


def setup_logger(log_file_path: str, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    try:
        os.makedirs(log_file_path)
    except FileExistsError:
        pass

    log_file_name = os.path.join(log_file_path, "logs.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
        if tag == "best_loss":
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
            elif type == "log":
                log_value = rng.uniform(space["lower"], space["upper"])
                params[key] = 10**log_value

        return params


def plot_2d(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    show_plot: bool = True,
    plot_lines: bool = False,
    save_path: str = None,
    loss: float = None,
    accuracy: float = None,
) -> None:
    predicted = predicted.detach().numpy()
    targets = targets.detach().numpy()
    plt.figure(figsize=(6, 4))
    plt.plot(
        targets[:, 0, 0],
        targets[:, 0, 1],
        "o",
        color="blue",
        markersize=1,
        label="targets",
    )
    plt.plot(
        predicted[:, 0, 0],
        predicted[:, 0, 1],
        "o",
        color="green",
        markersize=1,
        label="predicted",
    )
    plt.plot(
        targets[:, 1:, 0],
        targets[:, 1:, 1],
        "o",
        color="blue",
        markersize=0.2,
    )
    plt.plot(
        predicted[:, 1:, 0],
        predicted[:, 1:, 1],
        "o",
        color="green",
        markersize=0.2,
    )

    # connect points with lines
    if plot_lines:
        for i in range(targets.shape[0]):
            plt.plot(
                [targets[i, :, 0], predicted[i, :, 0]],
                [targets[i, :, 1], predicted[i, :, 1]],
                "r-",
                lw=0.05,
            )

    plt.legend(loc="upper right")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    if loss is not None:
        plt.title(f"Loss: {loss:.3e}, Accuracy: {accuracy:.2f}")
    if save_path is not None:
        plt.savefig(save_path + ".pdf")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_losses(
    seq_lens: np.ndarray,
    losses: np.ndarray,
    K: float,
    save_path: str = None,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(seq_lens, losses, "tab:blue")
    plt.xlabel("seq_len")
    plt.ylabel("Loss")
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path + ".pdf")
        plt.close()
    plt.show()


def plot_heat_map(
    predicted: np.ndarray,
    targets: np.ndarray,
    save_path: str = None,
    show_plot: bool = False,
) -> None:
    predicted = predicted.detach().numpy()
    targets = targets.detach().numpy()

    distances = np.sum((predicted - targets) ** 2, axis=-1)
    distances = np.log10(distances).T

    avg_distance = np.mean(distances, axis=-1)

    _, bins = np.histogram(distances, bins=20)
    distributions = [np.histogram(timestep, bins=bins)[0] for timestep in distances]

    plt.imshow(
        distributions,
        aspect="auto",
        origin="lower",
        extent=(bins[0], bins[-1], -0.5, distances.shape[0] - 0.5),
    )
    plt.plot(avg_distance, range(len(avg_distance)), "tab:red", lw=2)
    plt.xlabel("log10 mse")
    plt.ylabel("timestep")
    plt.colorbar(label="counts")
    plt.title("Squared errors")
    if save_path is not None:
        plt.savefig(save_path + ".pdf")
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_spatial_errors(
    predictions, targets, num_bins=20, power=2.5 / 10, save_path=None, show_plot=False
):
    x_true = predictions[:, :, 0].flatten()
    y_true = predictions[:, :, 1].flatten()
    x_pred = targets[:, :, 0].flatten()
    y_pred = targets[:, :, 1].flatten()

    x_bins = np.linspace(0, 1, num_bins + 1)
    y_bins = np.linspace(0, 1, num_bins + 1)

    errors = (x_true - x_pred) ** 2 + (y_true - y_pred) ** 2

    error_sum, _, _ = np.histogram2d(
        y_true, x_true, bins=[y_bins, x_bins], weights=errors
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(error_sum**power, origin="lower", extent=[0, 1, 0, 1])
    plt.colorbar(label=r"$\sim$Squared Errors")
    plt.xlabel(r"$\theta$")
    plt.ylabel("p")
    plt.title("Squared Errors")
    if save_path is not None:
        plt.savefig(save_path + ".pdf")
    if show_plot:
        plt.show()
    else:
        plt.close()


def conditional_torch_compile(compile, *args, **kwargs):
    def decorator(func):
        if compile:
            compiled_func = torch.compile(func, *args, **kwargs)

            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return compiled_func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

        return wrapped

    return decorator


def import_parsed_args(script_name: str) -> Namespace:
    parser = ArgumentParser(prog=script_name)

    parser.add_argument(
        "--path",
        type=str,
        help="Path to the experiment directory.",
    )

    if script_name == "Gridsearch step":
        parser.add_argument(
            "--default_params",
            "-default",
            action="store_true",
            help="Use default parameters for the gridsearch. (default: False)",
        )

    elif script_name == "Autoregressor trainer":
        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of epochs to train the model for.",
        )
        parser.add_argument(
            "--monitor_checkpoint",
            type=str,
            default="loss/train",
            help="Monitor value for checkpointing. (default: loss/train)",
        )
        parser.add_argument(
            "--mode_checkpoint",
            type=str,
            default="min",
            help="Mode for checkpointing. (default: min)",
        )
        parser.add_argument(
            "--monitor_stopping",
            type=str,
            default="loss/train",
            help="Monitor value for stopping. (default: loss/train)",
        )
        parser.add_argument(
            "--mode_stopping",
            type=str,
            default="min",
            help="Mode for stopping. (default: min)",
        )
        parser.add_argument(
            "--train_size",
            type=float,
            default=1.0,
            help="Fraction of data to use for training. (default: 1.0)",
        )
        parser.add_argument(
            "--progress_bar",
            "-prog",
            action="store_true",
            help="Show progress bar during training. (default: False)",
        )
        parser.add_argument(
            "--compile",
            action="store_true",
            help="Compile the model. (default: False)",
        )
        parser.add_argument(
            "--devices",
            type=int,
            default=1,
            help="Specify the number of devices to use. (default: 1)",
        )
        parser.add_argument(
            "--num_nodes",
            type=int,
            default=1,
            help="Specify number of nodes to use. (default: 1)",
        )
        parser.add_argument(
            "--checkpoint_path",
            "-ckpt",
            type=str,
            default=None,
            help="Path to the checkpoint file. (default: None)",
        )

    elif script_name == "Parameter updater":
        parser.add_argument(
            "--max_good_loss",
            type=float,
            default=1e-5,
            help="Maximum loss value considered acceptable for selecting parameters. (default: 1e-5)",
        )
        parser.add_argument(
            "--min_good_samples",
            type=int,
            default=4,
            help="Minimum number of good samples required to start updating parameters. (default: 4)",
        )
        parser.add_argument(
            "--check_every_n_steps",
            type=int,
            default=3,
            help="Check for new good samples every n steps. Its suggested that check_every_n_steps < min_good_samples, so that results are less likely to converge to a local optimium. (default: 3)",
        )
        parser.add_argument(
            "--current_step",
            type=int,
            default=1,
            help="Current step of the training. (default: 1)",
        )

    return parser.parse_args()
