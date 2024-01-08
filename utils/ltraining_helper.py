import torch.optim as optim
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class lModel(pl.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super(lModel, self).__init__()
        self.save_hyperparameters()

        self.hidden_size = kwargs.get("model_parameters").get("hidden_size")
        dropout_prob = kwargs.get("model_parameters").get("dropout_prob")
        self.num_layers = kwargs.get("model_parameters").get("num_layers")
        self.lr = kwargs.get("machine_learning_parameters").get("learn_rate")

        rnn_params = dict(dtype=torch.double)

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList(
            [torch.nn.RNNCell(2, self.hidden_size, **rnn_params)]
        )
        for layer in range(self.num_layers - 1):
            self.rnns.append(
                torch.nn.RNNCell(self.hidden_size, self.hidden_size, **rnn_params)
            )

        self.linear = torch.nn.Linear(self.hidden_size, 2, **rnn_params)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def _init_hidden(self, shape0: int, shape1: int):
        return [
            torch.zeros(shape0, shape1, dtype=torch.double)
            for layer in range(self.num_layers)
        ]

    def forward(self, input_t):
        outputs = []
        h_ts = self._init_hidden(input_t.shape[0], self.hidden_size)

        for input in input_t.split(1, dim=2):
            input = input.squeeze(2)

            for i in range(self.num_layers):
                if i == 0:
                    h_ts[i] = self.rnns[i](input, h_ts[i])
                    h_ts[i] = self.dropout(h_ts[i])
                    continue

                h_ts[i] = self.rnns[i](h_ts[i - 1], h_ts[i])
                if i < self.num_layers - 1:
                    h_ts[i] = self.dropout(h_ts[i])

            output = self.linear(h_ts[-1])
            outputs.append(output)

        return torch.stack(outputs, dim=2)

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams, {"metrics/final_val_loss": 0, "metrics/final_train_loss": 0}
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        inputs, outputs = batch
        predicted = self(inputs)[:, :, -1]
        loss = torch.nn.functional.mse_loss(predicted, outputs)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log("metrics/final_train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, outputs = batch
        predicted = self(inputs)[:, :, -1]
        loss = torch.nn.functional.mse_loss(predicted, outputs)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log("metrics/final_val_loss", loss)
        return loss


class Data:
    def __init__(self, map_object, plot_data: bool, **kwargs):
        map_object.generate_data()
        thetas, ps = map_object.retrieve_data()
        self.seq_len = kwargs.get("machine_learning_parameters").get("sequence_length")
        self.batch_size = kwargs.get("machine_learning_parameters").get("batch_size")

        if plot_data:
            map_object.plot_data()

        # preprocess_thetas
        thetas = thetas / np.pi - 1

        # data.shape = [init_points, 2, steps]
        self.data = np.stack([thetas.T, ps.T], axis=1)

    def prepare_data(
        self, shuffle: bool, t: float, plot_data_split: bool, input_size: float
    ):
        # first shuffle trajectories and then make sequences
        if shuffle:
            np.random.shuffle(self.data)
        # reduce dataset
        self.data = self.data[: int(len(self.data) * input_size)]
        if plot_data_split:
            self.plot_data_split(t)
        sequences = self._make_sequences()
        t = int(len(sequences) * t)
        self.train_loader = self._make_train_loader(sequences, t)
        self.val_loader = self._make_val_loader(sequences, t)

    def _make_train_loader(self, sequences, t):
        train_inputs = torch.from_numpy(sequences[:t, :, :-1]).clone()
        train_outputs = torch.from_numpy(sequences[:t, :, -1]).clone()
        train_tensor = Dataset(train_inputs, train_outputs)
        return DataLoader(
            train_tensor, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def _make_val_loader(self, sequences, t):
        val_inputs = torch.from_numpy(sequences[t:, :, :-1]).clone()
        val_outputs = torch.from_numpy(sequences[t:, :, -1]).clone()
        val_tensor = Dataset(val_inputs, val_outputs)
        return DataLoader(
            val_tensor, batch_size=2 * self.batch_size, shuffle=False, num_workers=4
        )

    def _make_sequences(self):
        init_points, features, steps = self.data.shape

        sequences = np.lib.stride_tricks.sliding_window_view(
            self.data, (1, features, self.seq_len + 1)
        )
        sequences = sequences.reshape(
            init_points * (steps - self.seq_len), features, self.seq_len + 1
        )
        # sequences.shape = [init_points * (steps - seq_len), features, seq_len + 1] as if we had more init_points
        # actually need seq_len + 1

        print(
            f"From '{init_points}' initial points and '{steps}' steps, we made '{sequences.shape[0]}' sequences of length '{self.seq_len}'."
        )

        return sequences

    def plot_data_split(self, t):
        t = int(len(self.data) * t)
        train_data = self.data[:t]
        val_data = self.data[t:]
        print(len(train_data), len(val_data))
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
        )  # for labels
        plt.plot(train_data[:, 0, 1:], train_data[:, 1, 1:], "bo", markersize=0.3)
        plt.plot(val_data[:, 0, 1:], val_data[:, 1, 1:], "ro", markersize=0.3)
        plt.legend()
        plt.show()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        x = x.clone().detach()
        y = y.clone().detach()
        return x, y


def return_elapsed_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        print(f"Elapsed time: {hours}h {minutes}m {seconds}s.")
    elif minutes > 0:
        print(f"Elapsed time: {minutes}m {seconds}s.")
    else:
        print(f"Elapsed time: {seconds}s.")


def plot_2d(predicted, outputs):
    predicted = predicted.detach().numpy()
    outputs = outputs.detach().numpy()
    plt.figure(figsize=(6, 4))
    plt.plot(outputs[:, 0, 0], outputs[:, 1, 0], "ro", markersize=2, label="targets")
    plt.plot(
        predicted[:, 0, 0],
        predicted[:, 1, 0],
        "bo",
        markersize=2,
        label="predicted",
    )
    plt.plot(outputs[:, 0, 1:], outputs[:, 1, 1:], "ro", markersize=0.5)
    plt.plot(predicted[:, 0, 1:], predicted[:, 1, 1:], "bo", markersize=0.5)
    plt.legend()
    plt.show()
