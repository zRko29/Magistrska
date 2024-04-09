from argparse import Namespace
import logging
from src.utils import (
    import_parsed_args,
    setup_logger,
    read_yaml,
    save_yaml,
)
import os
import numpy as np


class Gridsearch:
    def __init__(self, path: str, use_defaults: bool = False) -> None:
        self.path = path
        self.use_defaults = use_defaults

    def __next__(self):
        return self.update_params()

    def __iter__(self):
        for _ in range(10**3):
            yield self.update_params()

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




def main(args: Namespace, logger: logging.Logger) -> None:
    gridsearch = Gridsearch(args.params_dir, use_defaults=False)
    params = next(gridsearch)

    logger.info(f"{params=}")

    save_yaml(params, os.path.join(params["name"], "current_params.yaml"))


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Autoregressor trainer")

    args.params_dir = os.path.abspath(args.params_dir)

    params = read_yaml(args.params_dir)

    params["name"] = os.path.abspath(params["name"])

    logger = setup_logger(params["name"])
    logger.info("Started trainer.py")
    logger.info(f"{args.__dict__=}")

    main(args, logger)
