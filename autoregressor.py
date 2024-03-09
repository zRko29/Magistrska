import os, yaml

import pytorch_lightning as pl
from utils.mapping_helper import StandardMap
from utils.helper import Model, Data, plot_2d
from utils.dmd import DMD
from torch import Tensor

import warnings

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)
warnings.filterwarnings(
    "ignore",
    ".*across ranks is zero. Please make sure this was your intention*",
)


import logging

logging.getLogger("pytorch_lightning").setLevel(0)

if __name__ == "__main__":
    version: int | None = 1
    name: str = "overfitting_K=0.1"

    directory_path: str = f"logs/{name}"

    if version is not None:
        folders: list[str] = [os.path.join(directory_path, f"version_{version}")]
    else:
        folders: list[str] = [
            os.path.join(directory_path, folder)
            for folder in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, folder))
        ]
        folders.sort()

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        with open(params_path) as f:
            params: dict = yaml.safe_load(f)

        maps: list[StandardMap] = [
            StandardMap(seed=42, params=params),
            StandardMap(seed=41, params=params),
        ]
        input_suffixes: list[str] = ["standard", "random"]

        for map, input_suffix in zip(maps, input_suffixes):
            model_path: str = os.path.join(log_path, f"model.ckpt")
            model: Model = Model(**params).load_from_checkpoint(model_path)

            model.regression_seed = params["seq_length"]

            temp_params: dict = params.copy()
            temp_params.update({"seq_length": params.get("steps")})

            datamodule: Data = Data(
                map_object=map,
                train_size=1.0,
                plot_data=False,
                print_split=False,
                plot_data_split=False,
                params=temp_params,
            )

            trainer = pl.Trainer(
                precision=params["precision"],
                enable_progress_bar=False,
                logger=False,
            )
            predictions: dict = trainer.predict(model=model, dataloaders=datamodule)[0]
            predicted: Tensor = predictions["predicted"]
            targets: Tensor = predictions["targets"]
            loss: Tensor = predictions["loss"]

            print(f"{input_suffix} loss: {loss.item():.3e}")
            plot_2d(
                predicted,
                targets,
                show_plot=True,
                save_path=os.path.join(log_path, input_suffix),
                title=loss.item(),
            )

            dmd: DMD = DMD([predicted, targets])
            dmd.plot_source_matrix(titles=["Predicted", "Targets"])
            dmd._generate_dmd_results()
            dmd.plot_eigenvalues(titles=["Predicted", "Targets"])
            # dmd.plot_abs_values(titles=["Predicted", "Targets"])

        print("-" * 30)
