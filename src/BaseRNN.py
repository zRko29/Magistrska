import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from typing import Tuple, List
import pyprind

from src.custom_metrics import MSDLoss, PathAccuracy


class BaseRNN(pl.LightningModule):
    def __init__(self, **params):
        super(BaseRNN, self).__init__()
        self.save_hyperparameters()

        self.nonlin_hidden = params.get("nonlinearity_hidden")
        self.nonlin_lin = self.configure_non_linearity(params.get("nonlinearity_lin"))

        self.loss = self.configure_loss(params.get("loss"))
        self.accuracy = self.configure_accuracy("path_accuracy", 1.0e-5)

        self.num_rnn_layers: int = params.get("num_rnn_layers")
        self.num_lin_layers: int = params.get("num_lin_layers")
        self.sequence_type: str = params.get("sequence_type")
        self.lr: float = params.get("lr")
        self.optimizer: str = params.get("optimizer")

        # ------------------------------------------
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

    def _init_hidden(self, shape0: int, hidden_shapes: int) -> list[torch.Tensor]:
        return [
            torch.zeros(shape0, hidden_shape, device=self.device)
            for hidden_shape in hidden_shapes
        ]

    def configure_non_linearity(self, non_linearity: str) -> torch.nn.Module:
        if non_linearity is None:
            return torch.nn.Identity()
        elif non_linearity.lower() == "relu":
            return torch.nn.ReLU()
        elif non_linearity.lower() == "leaky_relu":
            return torch.nn.LeakyReLU()
        elif non_linearity.lower() == "tanh":
            return torch.nn.Tanh()
        elif non_linearity.lower() == "elu":
            return torch.nn.ELU()
        elif non_linearity.lower() == "selu":
            return torch.nn.SELU()

    def configure_accuracy(self, accuracy: str, threshold: float) -> torch.nn.Module:
        if accuracy == "path_accuracy":
            return PathAccuracy(threshold=threshold)

    def configure_loss(self, loss: str) -> torch.nn.Module:
        if loss in [None, "mse"]:
            return torch.nn.MSELoss()
        elif loss == "msd":
            return MSDLoss()
        elif loss == "rmse":
            return torch.nn.L1Loss()
        elif loss == "hubber":
            return torch.nn.SmoothL1Loss()

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

        if self.sequence_type == "many-to-one":
            predicted = predicted[:, -1:]

        loss, accuracy = self.compute_scores(predicted, targets)

        self.log_dict(
            {"loss/train": loss, "acc/train": accuracy},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, _) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)

        if self.sequence_type == "many-to-one":
            predicted = predicted[:, -1:]

        loss, accuracy = self.compute_scores(predicted, targets)

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
            # even with many-to-many training
            predicted_value = predicted_value[:, -1:]
            predicted = torch.cat([predicted, predicted_value], axis=1)

            pbar.update()

        predicted = predicted[:, self.regression_seed :]

        loss, accuracy = self.compute_scores(predicted, targets)

        return {
            "predicted": predicted,
            "targets": targets,
            "loss": loss,
            "accuracy": accuracy,
        }

    def compute_scores(
        self, predicted: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        loss = self.loss(predicted, targets)
        accuracy = self.accuracy(predicted, targets)
        return loss, accuracy

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
