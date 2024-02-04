import torch
import os, yaml
import numpy as np

from utils.mapping_helper import StandardMap
from utils.classification_helper import Model  # , plot_2d

if __name__ == "__main__":
    version = None
    name = "classification_1"

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
        with open(os.path.join(log_path, "hparams.yaml")) as f:
            params = yaml.safe_load(f)

        model_suffixes = [
            "",
            "-v1",
            "-v2",
        ]

        maps = [
            StandardMap(seed=42, params=params),
            StandardMap(seed=41, params=params),
        ]
        input_suffixes = ["standard", "random"]

        for map, input_suffix in zip(maps, input_suffixes):
            map.generate_data(lyapunov=True)
            thetas, ps = map.retrieve_data()
            spectrum = map.retrieve_spectrum()

            # data.shape = [init_points, 2, steps]
            data = np.stack([thetas.T, ps.T], axis=1)
            data = torch.from_numpy(data).to(torch.double)
            targets = torch.tensor(spectrum, dtype=torch.double)

            for model_suffix in model_suffixes:
                try:
                    model_path = os.path.join(log_path, f"lmodel{model_suffix}.ckpt")
                    model = Model(**params).load_from_checkpoint(model_path)
                except FileNotFoundError:
                    continue

                model.eval()
                with torch.inference_mode():
                    predicted = model(data).view(-1)

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted, targets
                )
                predictions = (predicted >= 0).int()
                accuracy = (predictions == targets).float().mean()

                print(f"{input_suffix}{model_suffix} loss: {loss.item():.3e}")
                print(f"{input_suffix}{model_suffix} accuracy: {accuracy.item():.3f}")

                # plot_2d(
                #     predicted,
                #     targets,
                #     show_plot=False,
                #     save_path=os.path.join(log_path, input_suffix + model_suffix),
                #     title=loss.item(),
                # )

        print("-" * 60)
