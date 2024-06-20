import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


from typing import List
import pyprind

try:
    from src.custom_metrics import MSDLoss, PathAccuracy
except ModuleNotFoundError:
    from custom_metrics import MSDLoss, PathAccuracy


class BaseRNN(pl.LightningModule):
    def __init__(self, **params):
        super(BaseRNN, self).__init__()
        self.save_hyperparameters()

        self.nonlin_hidden = params.get("nonlinearity_hidden")
        self.nonlin_lin = self.configure_non_linearity(params.get("nonlinearity_lin"))

        self.loss = self.configure_loss(params.get("loss"))
        self.accuracy = self.configure_accuracy("path_accuracy", 1.0e-4)

        self.num_rnn_layers: int = params.get("num_rnn_layers")
        self.num_lin_layers: int = params.get("num_lin_layers")
        self.lr: float = params.get("lr")
        self.optimizer: str = params.get("optimizer")

        # NOTE: This logic is for variable layer sizes
        hidden_sizes: List[int] = params.get("hidden_sizes")
        linear_sizes: List[int] = params.get("linear_sizes")

        rnn_layer_size: int = params.get("hidden_size")
        lin_layer_size: int = params.get("linear_size")

        self.hidden_sizes: List[int] = (
            hidden_sizes or [rnn_layer_size] * self.num_rnn_layers
        )
        self.linear_sizes: List[int] = linear_sizes or [lin_layer_size] * (
            self.num_lin_layers - 1
        )

    def create_linear_layers(self, compile: bool):
        self.lins = nn.ModuleList([])

        if self.num_lin_layers == 1:
            self.lins.append(nn.Linear(self.hidden_sizes[-1], 2))
        elif self.num_lin_layers > 1:
            self.lins.append(nn.Linear(self.hidden_sizes[-1], self.linear_sizes[0]))
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(
                    nn.Linear(self.linear_sizes[layer], self.linear_sizes[layer + 1])
                )
            self.lins.append(nn.Linear(self.linear_sizes[-1], 2))

        if compile:
            for layer in range(self.num_lin_layers):
                self.lins[layer] = torch.compile(self.lins[layer], dynamic=True)

    def _init_hidden(self, shape0: int, hidden_shapes: int) -> list[torch.Tensor]:
        return [
            torch.zeros(shape0, hidden_shape, device=self.device)
            for hidden_shape in hidden_shapes
        ]

    def configure_non_linearity(self, non_linearity: str) -> nn.Module:
        if non_linearity == "relu":
            return F.relu
        elif non_linearity == "leaky_relu":
            return F.leaky_relu
        elif non_linearity == "tanh":
            return F.tanh
        elif non_linearity == "elu":
            return F.elu
        elif non_linearity == "selu":
            return F.selu

    def configure_accuracy(self, accuracy: str, threshold: float) -> nn.Module:
        if accuracy == "path_accuracy":
            return PathAccuracy(threshold=threshold)

    def configure_loss(self, loss: str) -> nn.Module:
        if loss == "mse":
            return nn.MSELoss()
        elif loss == "msd":
            return MSDLoss()
        elif loss == "rmse":
            return nn.L1Loss()
        elif loss == "hubber":
            return nn.SmoothL1Loss()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        elif self.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, nesterov=True
            )

    def training_step(self, batch, _) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)
        predicted = predicted[:, -1:]

        targets = targets.to(self.dtype)
        loss = loss = self.loss(predicted, targets)

        self.log_dict({"loss/train": loss}, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, _) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch
        autoregression_seed = inputs.shape[1]
        autoregression_steps = targets.shape[1]

        for i in range(autoregression_steps):
            predicted_value = self(inputs[:, i:])
            predicted_value = predicted_value[:, -1:]
            inputs = torch.cat([inputs, predicted_value], axis=1)

        predicted = inputs[:, autoregression_seed:]

        targets = targets.to(self.dtype)
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
            iterations=batch.shape[1] - predicted.shape[1],
            bar_char="█",
            title="Predicting",
        )

        for i in range(batch.shape[1] - predicted.shape[1]):
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
        """
        Required to add best_loss to hparams in logger.
        """
        self._trainer.logger.log_hyperparams(self.hparams, {"best_loss": 1})

    def on_train_epoch_end(self):
        """
        Required to log best_loss at the end of the epoch. sync_dist=True is required to average the best_loss over all devices.
        """
        best_loss = self._trainer.callbacks[-1].best_model_score or 1
        self.log("best_loss", best_loss, sync_dist=True)


class ResidualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity):
        super(ResidualRNNCell, self).__init__()

        # Create the rnn cell
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity=nonlinearity)

        # Create the linear layer
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, input, hidden):
        candidate_hidden = self.rnn_cell(input, hidden)
        residual = self.linear(input)
        new_hidden = candidate_hidden + residual
        return new_hidden


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
        # Compute forget gate
        f_t = F.linear(h1, self.weight_fx, self.bias_f) + F.linear(h2, self.weight_fh)
        f_t = F.sigmoid(f_t)

        # Compute candidate activation
        h_hat_t = F.linear(h1, self.weight_hx, self.bias_h) + F.linear(
            f_t * h2, self.weight_hf
        )
        h_hat_t = F.tanh(h_hat_t)

        # Compute output
        h_t = (1 - f_t) * h2 + f_t * h_hat_t

        return h_t
