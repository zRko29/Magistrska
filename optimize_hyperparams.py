from argparse import Namespace
from joblib import Parallel, delayed

from trainer import main as train
from update_params import main as update

from src.helper import Gridsearch
from src.mapping_helper import StandardMap
from src.utils import measure_time, read_yaml, import_parsed_args, setup_logger


@measure_time
def main(args: Namespace, map_object: StandardMap) -> None:
    gridsearch = Gridsearch(args.params_dir, use_defaults=False)
    sleep_list = range(0, args.models_per_step * 5, 5)

    for i in range(args.optimization_steps):

        print()
        print(f"Starting optimization step: {i + 1} / {args.optimization_steps}")
        logger.info(f"Starting optimization step: {i + 1} / {args.optimization_steps}")
        print()

        try:
            Parallel(
                backend="loky",
                n_jobs=args.models_per_step,
                verbose=11,
            )(
                delayed(train)(args, params, sleep, map_object)
                for sleep, params in zip(sleep_list, gridsearch)
            )
        except Exception as e:
            logger.error(e)
            raise e

        update(args)
        print()
        print(f"Finished optimization step: {i + 1} / {args.optimization_steps}")
        logger.info(f"Finished optimization step: {i + 1} / {args.optimization_steps}")
        print()
        print("-----------------------------")


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Hyperparameter optimizer")

    params = read_yaml(args.params_dir)
    del params["gridsearch"]

    logs_dir = args.logs_dir or params["name"]

    logger = setup_logger(logs_dir)
    logger.info("Started optimize_hyperparams.py")
    logger.info(f"{args.__dict__=}")

    map_object = StandardMap(seed=42, params=params)

    main(args, map_object)
