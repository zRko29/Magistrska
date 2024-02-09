import torch.optim as optim
import pytorch_lightning as pl
import torch
import torchmetrics
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
        dropout = params.get("dropout")
        self.lr = params.get("lr")
        self.optimizer = params.get("optimizer")

        self.training_step_losses = []
        self.validation_step_losses = []
        self.training_step_accs = []
        self.validation_step_accs = []

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
        # h_ts[i].shape = [features, hidden_size]
        h_ts = self._init_hidden(input_t.shape[0], self.hidden_size)

        for input in input_t.split(1, dim=2):
            input = input.squeeze(2)

            # rnn layers
            h_ts[0] = self.rnns[0](input, h_ts[0])
            h_ts[0] = self.dropout(h_ts[0])
            for i in range(1, self.num_rnn_layers):
                h_ts[i] = self.rnns[i](h_ts[i - 1], h_ts[i])
                h_ts[i] = self.dropout(h_ts[i])

            # linear layers
            output = torch.relu(self.lins[0](h_ts[-1]))
            for i in range(1, self.num_lin_layers):
                output = self.lins[i](output)
                if i < self.num_lin_layers - 1:
                    output = torch.relu(output)

        # just take the last output
        return output

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
        loss = torch.nn.functional.cross_entropy(predicted, targets)
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets, task="binary"
        )

        self.log_dict(
            {"loss/train": loss, "acc/train": accuracy},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.training_step_losses.append(loss)
        self.training_step_accs.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predicted = self(inputs)
        loss = torch.nn.functional.cross_entropy(predicted, targets)
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets, task="binary"
        )

        self.log_dict(
            {"loss/val": loss, "acc/val": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.validation_step_losses.append(loss)
        self.validation_step_accs.append(accuracy)
        return loss

    def predict(self, batch, input_suffix):
        inputs, targets = batch
        predicted = self(inputs)
        loss = torch.nn.functional.cross_entropy(predicted, targets)
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets, task="binary"
        )

        print(f"{input_suffix} loss: {loss:.3e}")
        print(f"{input_suffix} accuracy: {accuracy:.3f}")


class Data(pl.LightningDataModule):
    def __init__(
        self,
        train_size: float,
        plot_data: bool,
        plot_data_split: bool,
        print_split: bool,
        params: dict,
        map_object=None,
        data_path=None,
    ):
        super(Data, self).__init__()
        if map_object is not None:
            map_object.generate_data(lyapunov=True)
            thetas, ps = map_object.retrieve_data()
            spectrum = map_object.retrieve_spectrum()

            if plot_data:
                map_object.plot_data()
        else:
            thetas = np.load(f"{data_path}/theta_values.npy")
            ps = np.load(f"{data_path}/p_values.npy")
            spectrum = np.load(f"{data_path}/spectrum.npy")

            thetas = thetas[: params.get("steps")]
            ps = ps[: params.get("steps")]
            spectrum = spectrum[: params.get("steps")]

        self.init_points = params.get("init_points")
        self.batch_size = params.get("batch_size")
        self.shuffle_paths = params.get("shuffle_paths")
        self.shuffle_batches = params.get("shuffle_batches")

        self.rng = np.random.default_rng(seed=42)

        # data.shape = [init_points, 2, steps]
        data = np.stack([thetas.T, ps.T], axis=1)

        # first shuffle trajectories and then make sequences
        if self.shuffle_paths:
            self.rng.shuffle(data)

        if plot_data_split:
            self.plot_data_split(data, train_size)

        xy_pairs = self._make_input_output_pairs(data, spectrum)

        t = int(len(xy_pairs) * train_size)
        self.train_data = xy_pairs[:t]
        self.val_data = xy_pairs[t:]

        if print_split:
            print(f"Sequences shape: {data.shape}")
            print(
                f"Train data shape: {len(self.train_data)} pairs of shape ({len(self.train_data[0][0][0])}, {1})"
            )
            if train_size < 1.0:
                print(
                    f"Validation data shape: {len(self.val_data)} pairs of shape ({len(self.val_data[0][0][0])}, {1})"
                )
            print()

        self.data = data
        self.spectrum = spectrum

    def _make_input_output_pairs(self, data, spectrum):
        return [
            (data[point], [1 - spectrum[point], spectrum[point]])
            for point in range(self.init_points)
        ]

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


class CustomCallback(pl.Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.min_train_loss = np.inf
        self.min_val_loss = np.inf
        self.max_train_acc = 0
        self.max_val_acc = 0

    def on_train_start(self, trainer, pl_module):
        trainer.logger.log_hyperparams(
            pl_module.hparams,
            {
                "metrics/min_val_loss": np.inf,
                "metrics/min_train_loss": np.inf,
                "metrics/max_val_acc": 0,
                "metrics/max_train_acc": 0,
            },
        )

    def on_train_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.training_step_losses).mean()
        if mean_loss < self.min_train_loss:
            self.min_train_loss = mean_loss
            pl_module.log("metrics/min_train_loss", mean_loss)
        mean_acc = torch.stack(pl_module.training_step_accs).mean()
        if mean_acc > self.max_train_acc:
            self.max_train_acc = mean_acc
            pl_module.log("metrics/max_train_acc", mean_acc)
        pl_module.training_step_losses.clear()
        pl_module.training_step_accs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.validation_step_losses).mean()
        if mean_loss < self.min_val_loss:
            self.min_val_loss = mean_loss
            pl_module.log("metrics/min_val_loss", mean_loss)
        mean_acc = torch.stack(pl_module.validation_step_accs).mean()
        if mean_acc > self.max_val_acc:
            self.max_val_acc = mean_acc
            pl_module.log("metrics/max_val_acc", mean_acc)
        pl_module.validation_step_losses.clear()
        pl_module.validation_step_accs.clear()

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
        with open(
            os.path.join(self.path, "classification_parameters.yaml"), "r"
        ) as file:
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
