import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

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
        self.rnn1 = torch.compile(nn.RNNCell(2, self.hidden_size), dynamic=False)
        self.rnn2 = torch.compile(
            MinimalGatedCell(self.hidden_size, self.hidden_size), dynamic=False
        )
        self.rnn3 = torch.compile(
            MinimalGatedCell(self.hidden_size, self.hidden_size), dynamic=False
        )

        # Create the linear layers
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
            h_ts2 = self.rnn2(h_ts1, h_ts2)
            h_ts3 = self.rnn3(h_ts2, h_ts3)
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

    def on_train_epoch_end(self):
        best_loss = self._trainer.callbacks[-1].best_model_score or 0
        self.log("best_acc", best_loss, sync_dist=True)


class MinimalGatedCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinimalGatedCell, self).__init__()

        # Parameters for forget gate
        self.weight_fx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_fh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_f = nn.Parameter(torch.Tensor(hidden_size))

        # Parameters for candidate activation
        self.weight_hx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_fx)
        nn.init.kaiming_uniform_(self.weight_fh)
        nn.init.zeros_(self.bias_f)

        nn.init.kaiming_uniform_(self.weight_hx)
        nn.init.kaiming_uniform_(self.weight_hf)
        nn.init.zeros_(self.bias_h)

    def forward(self, h1, h2):
        f_t = torch.sigmoid(
            F.linear(h1, self.weight_fx, self.bias_f) + F.linear(h2, self.weight_fh)
        )
        h_hat_t = torch.tanh(
            F.linear(h1, self.weight_hx, self.bias_h)
            + F.linear(f_t * h2, self.weight_hf)
        )
        h_t = (1 - f_t) * h2 + f_t * h_hat_t
        return h_t
