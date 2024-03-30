import torch.optim as optim
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import os
from typing import Tuple

from src.mapping_helper import StandardMap
from src.utils import read_yaml


class Model(pl.LightningModule):
    def __init__(
        self,
        **params: dict,
    ):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.num_rnn_layers: int = params.get("num_rnn_layers")
        self.num_lin_layers: int = params.get("num_lin_layers")
        self.sequence_type: str = params.get("sequence_type")
        dropout: float = params.get("dropout")
        self.lr: float = params.get("lr")
        self.optimizer: str = params.get("optimizer")

        # ----------------------
        # NOTE: This logic is kept so that variable layer sizes can be reimplemented in the future
        rnn_layer_size: int = params.get("hidden_size")
        lin_layer_size: int = params.get("linear_size")

        self.hidden_sizes: list[int] = [rnn_layer_size] * self.num_rnn_layers
        self.linear_sizes: list[int] = [lin_layer_size] * (self.num_lin_layers - 1)
        # ----------------------

        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        self.rnns.append(torch.nn.RNNCell(2, self.hidden_sizes[0]))
        for layer in range(self.num_rnn_layers - 1):
            self.rnns.append(
                torch.nn.RNNCell(self.hidden_sizes[layer], self.hidden_sizes[layer + 1])
            )

        # Create the linear layers
        self.lins = torch.nn.ModuleList([])
        if self.num_lin_layers == 1:
            self.lins.append(torch.nn.Linear(self.hidden_sizes[-1], 2))
        elif self.num_lin_layers > 1:
            self.lins.append(
                torch.nn.Linear(self.hidden_sizes[-1], self.linear_sizes[0])
            )
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(
                    torch.nn.Linear(
                        self.linear_sizes[layer], self.linear_sizes[layer + 1]
                    )
                )
            self.lins.append(torch.nn.Linear(self.linear_sizes[-1], 2))
        self.dropout = torch.nn.Dropout(p=dropout)

        # takes care of dtype
        self.to(torch.double)

    def _init_hidden(self, shape0: int, hidden_shapes: int) -> list[torch.Tensor]:
        return [
            torch.zeros(shape0, hidden_shape, dtype=torch.double).to(self.device)
            for hidden_shape in hidden_shapes
        ]

    def forward(self, input_t: torch.Tensor) -> torch.Tensor:
        outputs = []
        # h_ts[i].shape = [features, hidden_sizes]
        h_ts = self._init_hidden(input_t.shape[0], self.hidden_sizes)

        for input in input_t.split(1, dim=2):
            input = input.squeeze(2)

            # rnn layers
            h_ts[0] = self.rnns[0](input, h_ts[0])
            h_ts[0] = self.dropout(h_ts[0])
            for i in range(1, self.num_rnn_layers):
                h_ts[i] = self.rnns[i](h_ts[i - 1], h_ts[i])
                h_ts[i] = self.dropout(h_ts[i])

            # linear layers
            output = self.lins[0](h_ts[-1])
            for i in range(1, self.num_lin_layers):
                output = self.lins[i](output)

            outputs.append(output)

        return torch.stack(outputs, dim=2)

    def configure_optimizers(self) -> optim.Optimizer:
        if self.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        elif self.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch
        print(inputs.shape, targets.shape)

        predicted = self(inputs)
        if self.sequence_type == "many-to-one":
            predicted = predicted[:, :, -1:]
        loss = torch.nn.functional.mse_loss(predicted, targets)
        self.log(
            "loss/train",
            loss,
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)
        if self.sequence_type == "many-to-one":
            predicted = predicted[:, :, -1:]
        loss = torch.nn.functional.mse_loss(predicted, targets)
        self.log(
            "loss/val",
            loss,
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.validation_step_outputs.append(loss)
        return loss

    def predict_step(self, batch, batch_idx) -> dict[str, torch.Tensor]:
        predicted: torch.Tensor = batch[:, :, : self.regression_seed]
        targets: torch.Tensor = batch[:, :, self.regression_seed :]

        for i in range(batch.shape[2] - self.regression_seed):
            predicted_value = self(predicted[:, :, i:])[:, :, -1:]
            predicted = torch.cat([predicted, predicted_value], axis=2)

        predicted = predicted[:, :, self.regression_seed :]
        loss = torch.nn.functional.mse_loss(predicted, targets)

        return {"predicted": predicted, "targets": targets, "loss": loss}


class Data(pl.LightningDataModule):
    def __init__(
        self,
        map_object: StandardMap,
        train_size: float,
        params: dict,
        plot_data: bool = False,
        plot_data_split: bool = False,
    ) -> None:
        super(Data, self).__init__()
        self.seq_len: int = params.get("seq_length")
        self.batch_size: int = params.get("batch_size")
        self.shuffle_trajectories: bool = params.get("shuffle_trajectories")
        self.shuffle_batches: bool = params.get("shuffle_batches")
        self.shuffle_sequences: bool = params.get("shuffle_sequences")
        sequence_type: str = params.get("sequence_type")
        self.rng: np.random.Generator = np.random.default_rng(seed=42)

        map_object.generate_data()

        thetas: np.ndarray
        ps: np.ndarray
        thetas, ps = map_object.retrieve_data()

        if plot_data:
            map_object.plot_data()

        # data.shape = [init_points, 2, steps]
        self.data = np.stack([thetas.T, ps.T], axis=1)

        # shuffle trajectories
        if self.shuffle_trajectories:
            self.rng.shuffle(self.data)

        # many-to-many or many-to-one
        sequences = self._make_sequences(self.data, type=sequence_type)

        if plot_data_split:
            self.plot_data_split(sequences, train_size)

        self.input_output_pairs = self._make_input_output_pairs(
            sequences, type=sequence_type
        )
        self.t = int(len(self.input_output_pairs) * train_size)

    def _make_sequences(self, data: np.ndarray, type: str) -> np.ndarray:
        init_points: int
        features: int
        steps: int
        init_points, features, steps = data.shape

        if type == "many-to-many":
            # sequences.shape = [init_points*(steps//seq_len), 2, seq_len]
            sequences = np.split(data, steps // self.seq_len, axis=2)

            if not self.shuffle_sequences:
                sequences = np.array(
                    [seq[i] for i in range(init_points) for seq in sequences]
                )
            else:
                sequences = np.concatenate((sequences), axis=0)
                self.rng.shuffle(sequences)

        elif type == "many-to-one":
            if self.seq_len < steps:
                # sequences.shape = [init_points * (steps - seq_len), features, seq_len + 1]
                sequences = np.lib.stride_tricks.sliding_window_view(
                    data, (1, features, self.seq_len + 1)
                )
                sequences = sequences.reshape(
                    init_points * (steps - self.seq_len), features, self.seq_len + 1
                )
            elif self.seq_len == steps:
                sequences = data
        else:
            raise ValueError("Invalid type.")

        return sequences

    def _make_input_output_pairs(self, sequences, type: str) -> list[tuple[np.ndarray]]:
        # (trajectory, trajectory shifted by one step)
        if type == "many-to-many":
            return [(seq[:, :-1], seq[:, 1:]) for seq in sequences]
        elif type == "many-to-one":
            # (trajectory, next point in trajectory)
            return [(seq[:, :-1], seq[:, -1:]) for seq in sequences]
        else:
            raise ValueError("Invalid type.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.input_output_pairs[: self.t]),
            batch_size=self.batch_size,
            shuffle=self.shuffle_batches,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.input_output_pairs[self.t :]),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> torch.Tensor:
        return torch.tensor(self.data).to(torch.double).unsqueeze(0)

    def plot_data_split(self, dataset: torch.Tensor, train_ratio: float) -> None:
        train_size = int(len(dataset) * train_ratio)
        train_data = dataset[:train_size]
        val_data = dataset[train_size:]
        plt.figure(figsize=(6, 4))
        plt.plot(
            train_data[:, 0, 0],
            train_data[:, 1, 0],
            "bo",
            markersize=2,
            label="Training data",
        )
        plt.plot(
            val_data[:, 0, 0],
            val_data[:, 1, 0],
            "ro",
            markersize=2,
            label="Validation data",
        )
        plt.plot(train_data[:, 0, 1:], train_data[:, 1, 1:], "bo", markersize=0.3)
        plt.plot(val_data[:, 0, 1:], val_data[:, 1, 1:], "ro", markersize=0.3)
        plt.legend()
        plt.show()


class CustomCallback(pl.Callback):
    def __init__(self, print: bool) -> None:
        super(CustomCallback, self).__init__()
        self.min_train_loss = np.inf
        self.min_val_loss = np.inf

    def on_train_start(self, trainer, pl_module):
        trainer.logger.log_hyperparams(
            pl_module.hparams,
            {"metrics/min_val_loss": np.inf, "metrics/min_train_loss": np.inf},
        )

    def on_train_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.training_step_outputs).mean()
        if mean_loss < self.min_train_loss:
            self.min_train_loss = mean_loss
            pl_module.log(
                "metrics/min_train_loss",
                mean_loss,
                sync_dist=False,
            )
        pl_module.training_step_outputs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.validation_step_outputs).mean()
        if mean_loss < self.min_val_loss:
            self.min_val_loss = mean_loss
            pl_module.log(
                "metrics/min_val_loss",
                mean_loss,
                sync_dist=False,
            )
        pl_module.validation_step_outputs.clear()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[idx]
        x = torch.tensor(x).to(torch.double)
        y = torch.tensor(y).to(torch.double)
        return x, y


class Gridsearch:
    def __init__(self, path: str, use_defaults: bool = False) -> None:
        self.path = path
        self.use_defaults = use_defaults

    def __next__(self):
        return self.update_params()

    def __iter__(self):
        for _ in range(10**3):
            yield self.update_params()

    def update_params(self) -> dict:
        params = read_yaml(self.path)
        if not self.use_defaults:
            params = self._update_params(params)

        try:
            del params["gridsearch"]
        except KeyError:
            pass

        return params

    def _update_params(self, params) -> dict:
        # don't use any seed
        rng: np.random.Generator = np.random.default_rng(None)

        for key, space in params.get("gridsearch").items():
            type = space.get("type")
            if type == "int":
                params[key] = int(rng.integers(space["lower"], space["upper"] + 1))
            elif type == "choice":
                list = space.get("list")
                choice = rng.choice(list)
                try:
                    choice = float(choice)
                except:
                    choice = str(choice)
                params[key] = choice
            elif type == "float":
                params[key] = rng.uniform(space["lower"], space["upper"])

        return params


def plot_2d(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    show_plot: bool = True,
    save_path: str = None,
    title: str = None,
) -> None:
    predicted = predicted.detach().numpy()
    targets = targets.detach().numpy()
    plt.figure(figsize=(6, 4))
    plt.plot(targets[:, 0, 0], targets[:, 1, 0], "ro", markersize=2, label="targets")
    plt.plot(
        predicted[:, 0, 0],
        predicted[:, 1, 0],
        "bo",
        markersize=2,
        label="predicted",
    )
    plt.plot(targets[:, 0, 1:], targets[:, 1, 1:], "ro", markersize=0.5)
    plt.plot(predicted[:, 0, 1:], predicted[:, 1, 1:], "bo", markersize=0.5)
    plt.legend()
    if title is not None:
        plt.title(f"Loss = {title:.3e}")
    if save_path is not None:
        plt.savefig(save_path + ".pdf")
    if show_plot:
        plt.show()
    else:
        plt.close()
