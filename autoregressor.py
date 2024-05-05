import os
import pytorch_lightning as pl
from src.mapping_helper import StandardMap
from src.helper import Model, Data
from src.dmd import DMD
from src.utils import read_yaml, get_inference_folders, plot_2d, plot_heat_map
from typing import Optional, List
import warnings

warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
import logging

logging.getLogger("pytorch_lightning").setLevel(0)
pl.seed_everything(42, workers=True)


def main():
    version: Optional[int] = None
    directory_path: str = "logs/fitting_K=0.1"

    folders = get_inference_folders(directory_path, version)

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        params: dict = read_yaml(params_path)

        # params.update({"K": 0.1009})
        params.update(
            {
                "init_points": 40,
                "steps": 60,
            }
        )

        maps: List[StandardMap] = [
            StandardMap(seed=42, params=params),
            StandardMap(seed=41, params=params),
        ]
        input_suffixes: list[str] = ["standard", "random"]

        for map, input_suffix in zip(maps, input_suffixes):
            model_path: str = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path, map_location="cpu")

            # regression seed to take
            model.regression_seed = params["seq_length"]

            # how many steps to predict
            temp_params: dict = params.copy()
            temp_params.update({"seq_length": params.get("steps")})

            datamodule: Data = Data(
                map_object=map,
                train_size=1.0,
                params=temp_params,
            )

            trainer = pl.Trainer(
                precision=params["precision"],
                enable_progress_bar=False,
                logger=False,
                deterministic=True,
            )
            predictions: dict = trainer.predict(model=model, dataloaders=datamodule)[0]

            print(f"{input_suffix} loss: {predictions['loss'].item():.3e}")
            plot_2d(
                predictions["predicted"],
                predictions["targets"],
                show_plot=False,
                plot_lines=False,
                save_path=os.path.join(log_path, input_suffix),
                title=predictions["loss"].item(),
            )

            plot_heat_map(
                predictions["predicted"],
                predictions["targets"],
                save_path=os.path.join(log_path, input_suffix) + "_histogram",
            )

            # dmd: DMD = DMD([predictions["predicted"], predictions["targets"]])
            # dmd.plot_source_matrix(titles=["Predicted", "Targets"])
            # dmd._generate_dmd_results()
            # dmd.plot_eigenvalues(titles=["Predicted", "Targets"])
            # dmd.plot_abs_values(titles=["Predicted", "Targets"])

        print("-----------------------------")


if __name__ == "__main__":
    main()
