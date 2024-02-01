import torch.optim as optim
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import os, yaml


class Model(pl.LightningModule):
    def __init__(
        self,
        **params: dict,
    ):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.hidden_size = params.get("hidden_size")
        self.linear_size = params.get("linear_size")
        self.num_rnn_layers = params.get("num_rnn_layers")
        self.num_lin_layers = params.get("num_lin_layers")
        self.sequence_type = params.get("sequence_type")
        dropout = params.get("dropout")
        self.lr = params.get("lr")
        self.optimizer = params.get("optimizer")

        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        self.rnns.append(torch.nn.RNNCell(2, self.hidden_size))
        for layer in range(self.num_rnn_layers - 1):
            self.rnns.append(torch.nn.RNNCell(self.hidden_size, self.hidden_size))

        # Create the linear layers
        self.lins = torch.nn.ModuleList([])
        if self.num_lin_layers == 1:
            self.lins.append(torch.nn.Linear(self.hidden_size, 2))
        elif self.num_lin_layers > 1:
            self.lins.append(torch.nn.Linear(self.hidden_size, self.linear_size))
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(torch.nn.Linear(self.linear_size, self.linear_size))
            self.lins.append(torch.nn.Linear(self.linear_size, 2))
        self.dropout = torch.nn.Dropout(p=dropout)

        # takes care of dtype
        self.to(torch.double)

    def _init_hidden(self, shape0: int, shape1: int):
        return [
            torch.zeros(shape0, shape1, dtype=torch.double).to(self.device)
            for layer in range(self.num_rnn_layers)
        ]

    def forward(self, input_t):
        outputs = []
        # h_ts[i].shape = [features, hidden_size]
        h_ts = self._init_hidden(input_t.shape[0], self.hidden_size)

        for input in input_t.split(1, dim=2):
            input = input.squeeze(2)

            # rnn layers
            h_ts[0] = self.rnns[0](input, h_ts[0])
            h_ts[0] = self.dropout(h_ts[0])
            for i in range(1, self.num_rnn_layers):
                h_ts[i] = self.rnns[i](h_ts[i - 1], h_ts[i])
                if i < self.num_rnn_layers - 1:
                    h_ts[i] = self.dropout(h_ts[i])

            # linear layers
            output = self.lins[0](h_ts[-1])
            for i in range(1, self.num_lin_layers):
                output = self.lins[i](output)

            outputs.append(output)

        return torch.stack(outputs, dim=2)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        elif self.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
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

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predicted = self(inputs)
        if self.sequence_type == "many-to-one":
            predicted = predicted[:, :, -1:]
        loss = torch.nn.functional.mse_loss(predicted, targets)
        self.log("loss/val", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.validation_step_outputs.append(loss)
        return loss


class Data(pl.LightningDataModule):
    def __init__(
        self,
        map_object,
        train_size: float,
        if_plot_data: bool,
        if_plot_data_split: bool,
        params: dict,
    ):
        super(Data, self).__init__()
        map_object.generate_data()
        thetas, ps = map_object.retrieve_data()

        if if_plot_data:
            map_object.plot_data()

        self.seq_len = params.get("seq_length")
        self.batch_size = params.get("batch_size")
        self.shuffle_paths = params.get("shuffle_paths")
        self.shuffle_batches = params.get("shuffle_batches")
        self.shuffle_sequences = params.get("shuffle_sequences")
        sequence_type = params.get("sequence_type")

        self.rng = np.random.default_rng(seed=42)

        # preprocess data
        # thetas = thetas / np.pi - 1  # way 1
        thetas = thetas - np.pi  # way 2
        thetas /= np.max(np.abs(thetas))
        ps /= np.max(np.abs(ps))

        # data.shape = [init_points, 2, steps]
        data = np.stack([thetas.T, ps.T], axis=1)

        # first shuffle trajectories and then make sequences
        if self.shuffle_paths:
            self.rng.shuffle(data)

        # many-to-many or many-to-one types of sequences
        sequences = self._make_sequences(data, type=sequence_type)

        if if_plot_data_split:
            self.plot_data_split(sequences, train_size)

        print(f"Sequences shape: {sequences.shape}")
        xy_pairs = self._make_xy_pairs(sequences, type=sequence_type)

        t = int(len(xy_pairs) * train_size)
        self.train_data = xy_pairs[:t]
        self.val_data = xy_pairs[t:]

        print(
            f"Train data shape: {len(self.train_data)} pairs of shape ({len(self.train_data[0][0][0])}, {len(self.train_data[0][1][0])})"
        )
        if train_size < 1.0:
            print(
                f"Validation data shape: {len(self.val_data)} pairs of shape ({len(self.val_data[0][0][0])}, {len(self.val_data[0][1][0])})"
            )
        print()

    def _make_sequences(self, data, type: str):
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
            # sequences.shape = [init_points * (steps - seq_len), features, seq_len + 1]
            sequences = np.lib.stride_tricks.sliding_window_view(
                data, (1, features, self.seq_len + 1)
            )
            sequences = sequences.reshape(
                init_points * (steps - self.seq_len), features, self.seq_len + 1
            )
        else:
            raise ValueError("Invalid type.")

        return sequences

    def _make_xy_pairs(self, sequences, type: str):
        if type == "many-to-many":
            return [(seq[:, :-1], seq[:, 1:]) for seq in sequences]
        elif type == "many-to-one":
            return [(seq[:, :-1], seq[:, -1:]) for seq in sequences]
        else:
            raise ValueError("Invalid type.")

    def train_dataloader(self):
        return DataLoader(
            Dataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=self.shuffle_batches,
        )

    def val_dataloader(self):
        return DataLoader(
            Dataset(self.val_data),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )

    def plot_data_split(self, dataset, train_ratio):
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
    def __init__(self):
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
            pl_module.log("metrics/min_train_loss", mean_loss)
        pl_module.training_step_outputs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.validation_step_outputs).mean()
        if mean_loss < self.min_val_loss:
            self.min_val_loss = mean_loss
            pl_module.log("metrics/min_val_loss", mean_loss)
        pl_module.validation_step_outputs.clear()

    def on_fit_start(self, trainer, pl_module):
        print()
        print("Training started!")
        print()
        self.t_start = time.time()

    def on_fit_end(self, trainer, pl_module):
        print()
        print("Training ended!")
        train_time = time.time() - self.t_start
        print(f"Training time: {timedelta(seconds=train_time)}")
        print()
        # trainer.logger.experiment.add_scalar(
        #     "train_time", train_time, trainer.current_epoch
        # )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x).to(torch.double)
        y = torch.tensor(y).to(torch.double)
        return x, y


class Gridsearch:
    def __init__(self, path, num_vertices):
        self.path = path
        self.grid_step = 1
        self.num_vertices = num_vertices

    def get_params(self):
        with open(os.path.join(self.path, "classification_parameters.yaml"), "r") as file:
            params = yaml.safe_load(file)
            if self.num_vertices > 0:
                params = self._update_params(params)
                self.grid_step += 1
            if params.get("gridsearch") is not None:
                del params["gridsearch"]

        return params

    def _update_params(self, params):
        for key, space in params.get("gridsearch").items():
            if isinstance(space, dict):
                dtype = space["dtype"]
                if dtype == "int":
                    lower = space["lower"]
                    upper = space["upper"]
                    params[key] = np.random.randint(lower, upper + 1)
                elif dtype == "bool":
                    params[key] = np.random.choice([True, False])
                elif dtype == "float":
                    lower = space["lower"]
                    upper = space["upper"]
                    params[key] = np.random.uniform(lower, upper)
            print(f"{key} = {params[key]}")

        print("-" * 80)
        print(f"Gridsearch step: {self.grid_step} / {self.num_vertices}")
        print()

        return params


def plot_2d(predicted, targets, show_plot=True, save_path=None, title=None):
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
