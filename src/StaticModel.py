import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.model_summary import ModelSummary

import pyprind

try:
    from src.custom_metrics import MSDLoss, PathAccuracy
except ModuleNotFoundError:
    from custom_metrics import MSDLoss, PathAccuracy


class Model(pl.LightningModule):
    """
    The goal is to make model fully compilable.
    """

    def __init__(self, **params):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.seq_length = 100
        self.batch_size = 512
        self.hidden_size = 256
        self.linear_size = 256
        self.val_reg_preds = 80
        self.lr = 1.0e-5

        self.example_input_array: torch.Tensor = torch.randn(
            self.batch_size, self.seq_length, 2
        )

        self.loss = MSDLoss()
        self.accuracy = PathAccuracy(threshold=1.0e-4)

        # Create the RNN layers
        # self.rnn1 = nn.RNNCell(2, self.hidden_size)
        # self.rnn2 = HybridRNNCell(self.hidden_size)
        # self.rnn3 = HybridRNNCell(self.hidden_size)
        self.rnn1 = torch.compile(nn.RNNCell(2, self.hidden_size), dynamic=False)
        self.rnn2 = torch.compile(HybridRNNCell(self.hidden_size), dynamic=False)
        self.rnn3 = torch.compile(HybridRNNCell(self.hidden_size), dynamic=False)

        # Create the linear layers
        # self.lin1 = nn.Linear(self.hidden_size, self.linear_size)
        # self.bn1 = nn.BatchNorm1d(self.linear_size)
        # self.lin2 = nn.Linear(self.linear_size, self.linear_size)
        # self.bn2 = nn.BatchNorm1d(self.linear_size)
        # self.lin3 = nn.Linear(self.linear_size, 2)
        self.lin1 = torch.compile(
            nn.Linear(self.hidden_size, self.linear_size), dynamic=False
        )
        self.bn1 = torch.compile(nn.BatchNorm1d(self.linear_size), dynamic=False)
        self.lin2 = torch.compile(
            nn.Linear(self.linear_size, self.linear_size), dynamic=False
        )
        self.bn2 = torch.compile(nn.BatchNorm1d(self.linear_size), dynamic=False)
        self.lin3 = torch.compile(nn.Linear(self.linear_size, 2), dynamic=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        x = x.transpose(0, 1)

        h_ts1 = torch.zeros(self.batch_size, self.hidden_size, device=self.device)
        h_ts2 = torch.zeros(self.batch_size, self.hidden_size, device=self.device)
        h_ts3 = torch.zeros(self.batch_size, self.hidden_size, device=self.device)

        outputs = []
        # rnn layers
        for t in range(self.seq_length):
            h_ts1 = self.rnn1(x[t], h_ts1)
            h_ts2 = self.rnn2(x[t], h_ts2, h_ts1)
            h_ts3 = self.rnn3(x[t], h_ts3, h_ts2)
            outputs.append(h_ts3)

        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)

        # linear layers
        # 1
        outputs = self.lin1(outputs).transpose(1, 2)
        outputs = self.bn1(outputs).transpose(1, 2)
        outputs = torch.tanh(outputs)
        # 2
        outputs = self.lin2(outputs).transpose(1, 2)
        outputs = self.bn2(outputs).transpose(1, 2)
        outputs = torch.tanh(outputs)
        # 3
        outputs = self.lin3(outputs)

        return outputs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, _) -> torch.Tensor:
        inputs, targets = batch

        predicted = self(inputs)
        predicted = predicted[:, -1:]

        loss = self.loss(predicted, targets)

        self.log_dict({"loss/train": loss}, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, _) -> torch.Tensor:
        inputs, targets = batch

        for i in range(self.val_reg_preds):
            predicted_value = self(inputs[:, i:])
            predicted_value = predicted_value[:, -1:]
            inputs = torch.cat([inputs, predicted_value], axis=1)

        predicted = inputs[:, self.seq_length :]

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
        self._trainer.logger.log_hyperparams(self.hparams, {"best_acc": 0})
        # self._trainer.logger.log_hyperparams(self.hparams, {"best_loss": 1})

    def on_train_epoch_end(self):
        best_loss = self._trainer.callbacks[-1].best_model_score or 0
        self.log("best_acc", best_loss, sync_dist=True)
        # best_loss = self._trainer.callbacks[-1].best_model_score or 1
        # self.log("best_loss", best_loss, sync_dist=True)


class HybridRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super(HybridRNNCell, self).__init__()

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, 2))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))

        # hidden state at time t-1
        self.weight_hh1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh1 = nn.Parameter(torch.Tensor(hidden_size))

        # hidden state at previous layer
        self.weight_hh2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh2 = nn.Parameter(torch.Tensor(hidden_size))

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
    model = Model()
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
