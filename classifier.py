import torch
import os, yaml

from utils.mapping_helper import StandardMap
from utils.classification_helper import Model, Data  # , plot_2d

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
        with open(os.path.join(log_path, "hparams.yaml")) as f:
            params = yaml.safe_load(f)

        maps = [
            StandardMap(seed=42, params=params),
            StandardMap(seed=41, params=params),
        ]
        input_suffixes = ["standard", "random"]

        for map, input_suffix in zip(maps, input_suffixes):
            datamodule = Data(
                map_object=map,
                train_size=1.0,
                plot_data=False,
                plot_data_split=False,
                print_split=False,
                params=params,
            )

            dataloader = iter(datamodule.train_dataloader())

            model_path = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path)

            model.eval()
            with torch.inference_mode():
                model.predict(next(dataloader), input_suffix)

        print("-" * 60)
