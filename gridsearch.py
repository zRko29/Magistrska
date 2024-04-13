import os
from src.utils import (
    read_yaml,
    save_yaml,
    setup_logger,
    measure_time,
    Gridsearch,
)


@measure_time
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

    required_time = main(params_dir)

    logger.info(f"Finished gridsearch.py in {required_time}.")
