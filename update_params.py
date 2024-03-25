import yaml
from typing import Dict, Tuple, List
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from argparse import Namespace

INPUT_MAPPING = {"y": True, "n": False, "": False}
TYPES_LIST = ["float", "int", "choice"]

from src.utils import read_yaml, import_parsed_args, setup_logger


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
    try:
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
    except FileNotFoundError as e:
        logger.error(e)
        raise e

    return pd.DataFrame(all_loss_hyperparams)


def compute_parameter_intervals(
    results: pd.DataFrame, args: Namespace
) -> Dict[str, Tuple[float, float]]:
    gridsearch_params = read_yaml(args.params_dir)["gridsearch"]

    parameters = []
    for column in results.columns:
        if (
            len(results[column].unique()) > 1 or column in gridsearch_params.keys()
        ) and column != "best_loss":
            try:
                type = gridsearch_params[column]["type"]
            except KeyError:
                logger.warning(
                    f"Variable parameter '{column}' is not included in the gridsearch parameters."
                )
                continue
            if type:
                parameters.append(Parameter(name=column, type=type))

    # don't filter before because a parmeter could have the same value for all "good" rows
    # don't filter after because you wouldn't get optimal intervals
    try:
        results = results[results["best_loss"] < args.max_loss]
    except KeyError:
        logger.warning("There are probably no results in folder.")
        return None

    if len(results) < args.min_good_samples:
        logger.warning(
            f"Found {len(results)} (< {args.min_good_samples}) good samples. Parameters will not be updated."
        )
        return None
    else:
        logger.info(
            f"Found {len(results)} (> {args.min_good_samples}) good samples. Parameters will be updated."
        )

    for param in parameters:
        if param.type == "float":
            param.min = results[param.name].min()
            param.max = results[param.name].max()

        elif param.type == "int":
            param.min = results[param.name].min()
            param.max = results[param.name].max()

        elif param.type == "choice":
            param.value_counts = results[param.name].value_counts().to_dict()
            param.count = results[param.name].count()

            # only keep "good" values in list
            dict_copy = param.value_counts.copy()
            for key, value_count in param.value_counts.items():
                if value_count < param.count * 1 / 5 and len(dict_copy) > 1:
                    del dict_copy[key]
            param.value_counts = list(dict_copy.keys())

    return parameters


def update_yaml_file(
    args: Namespace, events_dir: str, parameters: List[Parameter]
) -> None:
    if parameters is not None:
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

        yaml_params = read_yaml(args.params_dir)

        # specify new folder
        yaml_params["name"] = find_new_path(events_dir)

        # update gridsearch parameters
        yaml_params["gridsearch"] = gridsearch_dict

        save_yaml(yaml_params, args.params_dir)
        save_last_params(yaml_params, events_dir)


def save_last_params(yaml_params: dict, events_dir: str) -> None:
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


def main(args: Namespace) -> None:
    events_dir = read_yaml(args.params_dir)["name"]

    loss_and_params = get_loss_and_params(events_dir)
    parameters = compute_parameter_intervals(results=loss_and_params, args=args)

    update_yaml_file(args, events_dir, parameters)


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Parameter updater")

    params = read_yaml(args.params_dir)

    logs_dir = args.logs_dir or params["name"]

    logger = setup_logger(logs_dir)
    logger.info("Started update_params.py")
    logger.info(f"{args.__dict__=}")

    main(args)
