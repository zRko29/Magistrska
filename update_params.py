import yaml
from typing import Dict, Tuple, List
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
from math import ceil

import pandas as pd

INPUT_MAPPING = {"y": True, "n": False, "": False}
TYPES_LIST = ["float", "int", "choice"]


class Parameter:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

        if self.type in ["float", "int"]:
            self.min = float("inf")
            self.max = -float("inf")

        elif self.type == "choice":
            self.value_counts = {}
            self.count = 0


def read_yaml(param_file_path: str) -> dict[str | float | int]:
    with open(param_file_path, "r") as param_file:
        return yaml.safe_load(param_file)


def save_yaml(file: dict, param_file_path: str) -> dict[str | float | int]:
    with open(param_file_path, "w") as f:
        yaml.safe_dump(file, f, default_flow_style=None, default_style=None)


def read_events_file(events_file_path: str) -> EventAccumulator:
    event_acc = EventAccumulator(events_file_path)
    event_acc.Reload()
    return event_acc


def extract_best_loss_from_event_file(events_file_path: str) -> str | float | int:
    event_values = read_events_file(events_file_path)
    for tag in event_values.Tags()["scalars"]:
        if tag == "metrics/min_train_loss":
            return {"best_loss": event_values.Scalars(tag)[-1].value}


def get_loss_and_params(dir: str) -> pd.DataFrame:
    all_loss_hyperparams = []
    for directory in sorted(os.listdir(dir)):
        loss_value = None
        parameter_dict = None
        if os.path.isdir(os.path.join(dir, directory)):
            for file in os.listdir(os.path.join(dir, directory)):
                if "events" in file.split("."):
                    file_path = os.path.join(dir, directory, file)
                    loss_value = extract_best_loss_from_event_file(file_path)

                elif file == "hparams.yaml":
                    file_path = os.path.join(dir, directory, file)
                    parameter_dict = read_yaml(file_path)

            if loss_value and parameter_dict:
                all_loss_hyperparams.append({**loss_value, **parameter_dict})

    return pd.DataFrame(all_loss_hyperparams)


def input_value(param: str, value: str, include: bool = False):
    if not include:
        include = INPUT_MAPPING[
            input(
                f"Parameter '{param}' was is not included in the gridsearch parameters. Do you want to include it now? (y/n): "
            )
        ]
    if include:
        type = input(f"Please choose value for {value} (int/float/choice): ")
        if type not in TYPES_LIST:
            print("Invalid type. Try again and please choose valid type.")
            type = input_value(param, value, include)
        return type
    return None


def compute_parameter_intervals(
    results: pd.DataFrame, params_dir: str, max_loss: float, min_n_good_values: int
) -> Dict[str, Tuple[float, float]]:
    gridsearch_params = read_yaml(params_dir)["gridsearch"]

    parameters = []
    for column in results.columns:
        if (
            len(results.get(column, {}).unique()) > 1
            or column in gridsearch_params.keys()
        ) and column != "best_loss":
            try:
                type = gridsearch_params[column]["type"]
            except KeyError:
                type = input_value(column, "type")
            if type:
                parameters.append(Parameter(name=column, type=type))

    # don't filter before because a parmeter could have the same value for all "good" rows
    # don't filter after because you wouldn't get optimal intervals
    try:
        results = results[results["best_loss"] < max_loss]
    except KeyError:
        print()
        print("There are probably no results in folder.")
        print()
        return None

    if len(results) < min_n_good_values:
        return None

    for param in parameters:
        if param.type == "float":
            param.min = results[param.name].min() * 0.99
            param.max = results[param.name].max() * 1.01

        elif param.type == "int":
            # use math.ceil to prevent zeros
            param.min = ceil(results[param.name].min() * 0.99)
            param.max = ceil(results[param.name].max() * 1.01)

        elif param.type == "choice":
            param.value_counts = results[param.name].value_counts().to_dict()
            param.count = results[param.name].count()

            # only keep "good" values
            dict_copy = param.value_counts.copy()
            for key, value_count in param.value_counts.items():
                if value_count < param.count * 1 / 5 and len(dict_copy) > 1:
                    del dict_copy[key]
            param.value_counts = list(dict_copy.keys())

    return parameters


def print_parameter_intervals(parameters: List[Parameter]):
    print()
    print("Parameter intervals:")
    print()
    for param in parameters:
        print(f"Parameter: {param.name}")
        if param.type in ["float", "int"]:
            if param.name == "lr":
                print(f"min: {param.min:.3e} || " f"max: {param.max:.3e}")
            else:
                print(f"min: {param.min} || " f"max: {param.max}")
        elif param.type == "choice":
            print(f"values: {param.value_counts}")
        print()


def update_yaml_file(
    params_dir: str, events_dir: str, parameters: List[Parameter], force: bool = False
):
    if parameters is not None:
        update = "y"
        if not force:
            update = INPUT_MAPPING[
                input("Do you want to update the parameters.yaml file? (y/n): ")
            ]

        if update:
            yaml_params = read_yaml(params_dir)
            new_path = find_new_path(events_dir)

            yaml_params["name"] = new_path

            gridsearch_dict = {}

            for param in parameters:
                if param.type in ["float", "int"]:
                    gridsearch_dict[param.name] = {
                        "lower": param.min,
                        "upper": param.max,
                        "type": param.type,
                    }
                elif param.type == "choice":
                    gridsearch_dict[param.name] = {
                        "list": param.value_counts,
                        "type": param.type,
                    }

            yaml_params["gridsearch"] = gridsearch_dict

            save_yaml(yaml_params, "config/auto_parameters.yaml")

            save_last_params(yaml_params, events_dir)


def save_last_params(yaml_params: dict, events_dir: str):
    folder = "/".join(events_dir.split("/")[:-1])
    save_yaml(yaml_params, os.path.join(folder, "last_parameters.yaml"))


def find_new_path(file_dir: str) -> str:
    path_split = file_dir.split("/")
    path_split[-1] = str(int(path_split[-1]) + 1)
    new_path = "/".join(path_split)
    try:
        os.mkdir(new_path)
    except FileExistsError:
        pass
    return new_path


if __name__ == "__main__":
    params_dir = "config/auto_parameters.yaml"
    events_dir = read_yaml(params_dir)["name"]

    loss_and_params = get_loss_and_params(events_dir)
    parameters = compute_parameter_intervals(
        loss_and_params, params_dir, max_loss=1e-6, min_n_good_values=3
    )
    # print_parameter_intervals(parameters)

    update_yaml_file(params_dir, events_dir, parameters, force=True)
