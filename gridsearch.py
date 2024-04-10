from argparse import Namespace
from src.utils import (
    import_parsed_args,
    setup_logger,
    read_yaml,
    save_yaml,
)
import os
import numpy as np


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


def main(params_dir: str) -> None:
    params_path = os.path.join(params_dir, "parameters.yaml")
    gridsearch = Gridsearch(params_path, use_defaults=False)
    updated_params = gridsearch.update_params()

    save_yaml(updated_params, os.path.join(params_dir, "current_params.yaml"))


if __name__ == "__main__":
    params_dir = os.path.abspath("config")

    params_path = os.path.join(params_dir, "parameters.yaml")
    params = read_yaml(params_path)

    params["name"] = os.path.abspath(params["name"])

    logger = setup_logger(params["name"])
    logger.info("Running gridsearch.py")

    main(params_dir)
