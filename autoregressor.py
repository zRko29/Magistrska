import os, yaml

import pytorch_lightning as pl
from utils.mapping_helper import StandardMap
from utils.helper import Model, Data, plot_2d
from utils.dmd import DMD

import warnings

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

import logging

logging.getLogger("pytorch_lightning").setLevel(0)

if __name__ == "__main__":
    version = None
    name = "overfitting_1"

    directory_path = f"logs/{name}"

    if version is not None:
        folders = [os.path.join(directory_path, f"version_{version}")]
    else:
        folders = [
            os.path.join(directory_path, folder)
            for folder in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, folder))
        ]
        folders.sort()

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path = os.path.join(log_path, "hparams.yaml")
        with open(params_path) as f:
            params = yaml.safe_load(f)

        maps = [
            StandardMap(seed=42, params=params),
            StandardMap(seed=41, params=params),
        ]
        input_suffixes = ["standard", "random"]

        for map, input_suffix in zip(maps, input_suffixes):
            model_path = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path)

            model.regression_seed = params["seq_length"]

            temp_params = params.copy()
            temp_params.update({"seq_length": params.get("steps")})

            datamodule = Data(
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
            predictions = trainer.predict(model=model, dataloaders=datamodule)[0]
            predicted = predictions["predicted"]
            targets = predictions["targets"]
            loss = predictions["loss"]

            print(f"{input_suffix} loss: {loss.item():.3e}")
            plot_2d(
                predicted,
                targets,
                show_plot=True,
                save_path=os.path.join(log_path, input_suffix),
                title=loss.item(),
            )

            dmd = DMD([predicted, targets])
            dmd.plot_source_matrix(titles=["Predicted", "Targets"])
            dmd._generate_dmd_results()
            dmd.plot_eigenvalues(titles=["Predicted", "Targets"])
            dmd.plot_abs_values(titles=["Predicted", "Targets"])

        print("-" * 30)
