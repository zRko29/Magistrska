import os, yaml
import pytorch_lightning as pl

import warnings

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

from utils.mapping_helper import StandardMap
from utils.classification_helper import Model, Data

if __name__ == "__main__":
    version = None
    name = "classification_1"

    directory_path = f"logs/{name}"

    if version is not None:
        folders = [os.path.join(directory_path, f"version_{str(version)}")]
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
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        params.update({"init_points": 30, "steps": 60, "sampling": "random"})
        maps = [
            StandardMap(K=1.5, params=params, seed=42),
            StandardMap(K=0.2, params=params, seed=41),
            StandardMap(K=1.0, params=params, seed=40),
            StandardMap(K=1.9, params=params, seed=39),
        ]

        for map in maps:
            datamodule = Data(
                map_object=map,
                train_size=1.0,
                plot_data=False,
                print_split=False,
                params=params,
            )

            model_path = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path)

            trainer = pl.Trainer()
            trainer.predict(model=model, dataloaders=datamodule)

            print("-" * 30)
