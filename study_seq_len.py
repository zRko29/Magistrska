import os
import pytorch_lightning as pl
from src.mapping_helper import StandardMap
from src.helper import Model, Data
from src.utils import read_yaml, get_inference_folders, plot_losses
from typing import Optional
import warnings
import numpy as np
import pyprind

warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
import logging

logging.getLogger("pytorch_lightning").setLevel(0)
pl.seed_everything(42, workers=True)


def main():
    version: Optional[int] = 0
    name: str = "overfitting_K=2.0"

    directory_path: str = f"logs/{name}"

    folders = get_inference_folders(directory_path, version)

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        params: dict = read_yaml(params_path)

        map = StandardMap(seed=42, params=params)

        seq_lens = np.arange(3, 50, 1)
        losses = []

        pbar = pyprind.ProgBar(iterations=len(seq_lens), bar_char="█", track_time=False)

        for seq_len in seq_lens:
            model_path: str = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path, map_location="cpu")

            # regression seed to take
            model.regression_seed = seq_len

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

            losses.append(predictions["loss"].item())
            pbar.update()

        plot_losses(
            seq_lens,
            losses,
            params.get("K"),
            # save_path=f"{log_path}/losses_K={params.get('K')}",
        )


if __name__ == "__main__":
    main()
