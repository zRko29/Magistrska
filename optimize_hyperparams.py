from argparse import ArgumentParser, Namespace

from trainer import main as train
from update_params import main as update

from src.helper import Gridsearch
from src.utils import measure_time


@measure_time
def main(args: Namespace, params: dict) -> None:
    for i in range(args.optimization_steps):
        print()
        print(f"Starting optimization step: {i + 1} / {args.optimization_steps}")
        print()

        for j in range(args.models_per_step):
            model_trained = j + 1
            total_model_trained = i * args.models_per_step + j + 1
            print(
                f"Training model: {model_trained} / {args.models_per_step} (total: {total_model_trained} / {args.optimization_steps*args.models_per_step})"
            )
            print()
            train(args, params)

        update(args)
        print()
        print("Finished optimization step.")
        print()
        print("------------------------")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Hyperparameter optimizer",
        description="A tool for optimizing hyperparameters, updating parameter files, and training autoregression models.",
    )

    # Optimization settings
    optimization_group = parser.add_argument_group("Optimization settings")
    optimization_group.add_argument(
        "--optimization_steps",
        type=int,
        default=5,
        help="Number of optimization steps to perform. (default: %(default)s)",
    )
    optimization_group.add_argument(
        "--models_per_step",
        type=int,
        default=5,
        help="Number of models to train in each optimization step. (default: %(default)s)",
    )

    # Update parameters settings
    update_group = parser.add_argument_group("Parameter update settings")
    update_group.add_argument(
        "--params_dir",
        type=str,
        default="config/auto_parameters.yaml",
        help="Directory containing parameter files. (default: %(default)s)",
    )
    update_group.add_argument(
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

    # Training settings
    training_group = parser.add_argument_group("Training settings")
    training_group.add_argument(
        "--progress_bar",
        "-prog",
        action="store_true",
        help="Display a progress bar during training. (default: %(default)s)",
    )
    training_group.add_argument(
        "--accelerator",
        "-acc",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Select the accelerator (auto, cpu, or gpu) for training. (default: %(default)s)",
    )
    training_group.add_argument(
        "--num_devices",
        default="auto",
        help="Number of devices to use for training. (default: %(default)s)",
    )
    training_group.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "ddp_spawn"],
        help="Select the training strategy (auto, ddp, or ddp_spawn). (default: %(default)s)",
    )

    args = parser.parse_args()

    gridsearch = Gridsearch(args.params_dir, use_defaults=True)
    params: dict = gridsearch.update_params()

    main(args, params)
