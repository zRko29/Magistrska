import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

import pyprind

from src.custom_metrics import MSDLoss, PathAccuracy


class Model(pl.LightningModule):
    """
    The goal is to make model fully compilable. This model uses specifically these hyperparameters:

    hidden_size: 256
    linear_size: 256
    nonlinearity_hidden: tanh
    nonlinearity_lin: tanh
    num_lin_layers: 3
    num_rnn_layers: 3
    rnn_type: hybrid

    batch_size: 256
    seq_length: 20

    optimizer: adam
    lr: 2.0e-05

    loss: msd
    sequence_type: many-to-one
    """

    def __init__(self, **params):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.loss = MSDLoss()
        self.accuracy = PathAccuracy(threshold=1.0e-5)

        # Create the RNN layers
        # self.rnn1 = nn.RNNCell(2, 256)
        # self.rnn2 = HybridRNNCell()
        # self.rnn3 = HybridRNNCell()
        self.rnn1 = torch.compile(nn.RNNCell(2, 256), dynamic=False)
        self.rnn2 = torch.compile(HybridRNNCell(), dynamic=False)
        self.rnn3 = torch.compile(HybridRNNCell(), dynamic=False)

        # Create the linear layers
        # self.lins1 = nn.Linear(256, 256)
        # self.bn_lins1 = nn.BatchNorm1d(256)
        # self.lins2 = nn.Linear(256, 256)
        # self.bn_lins2 = nn.BatchNorm1d(256)
        # self.lins3 = nn.Linear(256, 2)
        self.lins1 = torch.compile(nn.Linear(256, 256), dynamic=False)
        self.bn_lins1 = torch.compile(nn.BatchNorm1d(256), dynamic=False)
        self.lins2 = torch.compile(nn.Linear(256, 256), dynamic=False)
        self.bn_lins2 = torch.compile(nn.BatchNorm1d(256), dynamic=False)
        self.lins3 = torch.compile(nn.Linear(256, 2), dynamic=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        x = x.transpose(0, 1)

        h_ts1 = torch.zeros(256, 256, device=self.device)
        h_ts2 = torch.zeros(256, 256, device=self.device)
        h_ts3 = torch.zeros(256, 256, device=self.device)

        outputs = []
        # rnn layers
        for t in range(20):
            h_ts1 = self.rnn1(x[t], h_ts1)
            h_ts2 = self.rnn2(x[t], h_ts2, h_ts1)
            h_ts3 = self.rnn2(x[t], h_ts3, h_ts2)
            outputs.append(h_ts3)

        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)

        # linear layers
        # 1
        outputs = self.lins1(outputs).transpose(1, 2)
        outputs = self.bn_lins1(outputs).transpose(1, 2)
        outputs = torch.tanh(outputs)
        # 2
        outputs = self.lins2(outputs).transpose(1, 2)
        outputs = self.bn_lins2(outputs).transpose(1, 2)
        outputs = torch.tanh(outputs)
        # 3
        outputs = self.lins3(outputs)

        return outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=2.0e-5, amsgrad=True)

    def training_step(self, batch, _) -> torch.Tensor:
        inputs, targets = batch

        predicted = self(inputs)
        predicted = predicted[:, -1:]

        loss = self.loss(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        self.log_dict(
            {"loss/train": loss, "acc/train": accuracy},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, _) -> torch.Tensor:
        inputs, targets = batch

        predicted = self(inputs)
        predicted = predicted[:, -1:]

        loss = self.loss(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        self.log_dict(
            {"loss/val": loss, "acc/val": accuracy},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        return loss

    def predict_step(self, batch, _) -> dict[str, torch.Tensor]:
        predicted: torch.Tensor = batch[:, : self.regression_seed]
        targets: torch.Tensor = batch[:, self.regression_seed :]

        pbar = pyprind.ProgBar(
            iterations=batch.shape[1] - self.regression_seed,
            bar_char="█",
            title="Predicting",
        )

        for i in range(batch.shape[1] - self.regression_seed):
            predicted_value = self(predicted[:, i:])
            predicted_value = predicted_value[:, -1:]
            predicted = torch.cat([predicted, predicted_value], axis=1)

            pbar.update()

        predicted = predicted[:, self.regression_seed :]

        loss = self.loss(predicted, targets)
        accuracy = self.accuracy(predicted, targets)

        return {
            "predicted": predicted,
            "targets": targets,
            "loss": loss,
            "accuracy": accuracy,
        }

    @rank_zero_only
    def on_train_start(self):
        self._trainer.logger.log_hyperparams(self.hparams, {"best_loss": 1})

    def on_train_epoch_end(self):
        best_loss = self._trainer.callbacks[-1].best_model_score or 1
        self.log("best_loss", best_loss, sync_dist=True)


class HybridRNNCell(nn.Module):
    def __init__(self):
        super(HybridRNNCell, self).__init__()

        self.weight_ih = nn.Parameter(torch.Tensor(256, 2))
        self.bias_ih = nn.Parameter(torch.Tensor(256))

        # hidden state at time t-1
        self.weight_hh1 = nn.Parameter(torch.Tensor(256, 256))
        self.bias_hh1 = nn.Parameter(torch.Tensor(256))

        # hidden state at previous layer
        self.weight_hh2 = nn.Parameter(torch.Tensor(256, 256))
        self.bias_hh2 = nn.Parameter(torch.Tensor(256))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih)
        nn.init.zeros_(self.bias_ih)

        nn.init.kaiming_uniform_(self.weight_hh1)
        nn.init.zeros_(self.bias_hh1)

        nn.init.kaiming_uniform_(self.weight_hh2)
        nn.init.zeros_(self.bias_hh2)

    def forward(self, input, hidden1, hidden2):
        h_t = torch.tanh(
            nn.functional.linear(input, self.weight_ih, self.bias_ih)
            + nn.functional.linear(hidden1, self.weight_hh1, self.bias_hh1)
            + nn.functional.linear(hidden2, self.weight_hh2, self.bias_hh2)
        )
        return h_t


if __name__ == "__main__":
    from custom_metrics import MSDLoss, PathAccuracy

    model = Model()
    compiled_model = torch.compile(model)
    print(compiled_model(torch.randn(256, 20, 2)))
