import torch
import os, yaml
import numpy as np

from utils.mapping_helper import StandardMap
from utils.ltraining_helper import lModel, plot_2d

ROOT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(ROOT_DIR, "config")


log_path = "logs/masters/1704712903"

with open(os.path.join(log_path, "hparams.yaml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

model_path = os.path.join(log_path, "lmodel.ckpt")

model = lModel(**params).load_from_checkpoint(model_path)
map = StandardMap(init_points=5, steps=100, sampling="random")

model.eval()
with torch.inference_mode():
    map.generate_data()
    thetas, ps = map.retrieve_data()
    regression_seed = params.get("machine_learning_parameters").get(
        "sequence_length"
    )  # seems to work best

    # preprocess_thetas
    thetas = thetas / np.pi - 1
    # data.shape = [init_points, 2, steps]
    data = np.stack([thetas.T, ps.T], axis=1)

    assert (
        data.shape[2] > regression_seed
    ), "regression_seed must be smaller than the number of steps"

    predicted = torch.from_numpy(data[:, :, :regression_seed]).clone()
    outputs = torch.from_numpy(data).clone()

    model.eval()
    for i in range(data.shape[2] - regression_seed):
        predicted_value = model(predicted[:, :, i:])
        predicted_value = predicted_value[:, :, -1:]

        predicted = torch.cat([predicted, predicted_value], axis=2)

    predicted = predicted[:, :, regression_seed:]
    outputs = outputs[:, :, regression_seed:]

    loss = torch.nn.functional.mse_loss(predicted, outputs)
    print(f"Loss: {loss.item():.3e}")

plot_2d(predicted, outputs)
