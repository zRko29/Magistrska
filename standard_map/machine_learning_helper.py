import numpy as np
from tqdm import tqdm
from helper_functions import validate_data_type, HelperClass
from sklearn.model_selection import train_test_split
import os
import yaml
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

MAIN_DIR = os.path.dirname(__file__)

with open(os.path.join(MAIN_DIR, "config_stdm.yaml"), "r") as file:
    PARAMETERS = yaml.safe_load(file)


class Training(HelperClass):
    """
    A class for training a machine learning model.

    Attributes:
    -----------
    thetas : numpy.ndarray
        An array of shape (n_steps, n_features) containing the values of thetas.
    ps : numpy.ndarray
        An array of shape (n_steps, n_features) containing the values of ps.
    model : torch.nn.Module
        A PyTorch model to be trained.
    parameters : dict, optional
        A dictionary containing the hyperparameters for training the model.
    window_size : int, optional
        The size of the sliding window used to create the input sequences.
    step_size : int, optional
        The step size used to create the input sequences.
    train_size : float, optional
        The proportion of the data to be used for training.
    epochs : int, optional
        The number of epochs to train the model.
    batch_size : int, optional
        The batch size used for training.
    loss : str, optional
        The loss function used for training.
    learn_rate : float, optional
        The learning rate used for training.
    """
    def __init__(self, thetas, ps, model, parameters: dict = None, window_size: int = None, step_size: int = None, train_size: float = None, epochs: int = None, batch_size: int = None, loss: str = None, learn_rate: float = None):
        params = PARAMETERS.get("machine_learning_parameters")

        validate_data_type(params, dict, error_prefix="Config parameters dictionary")

        if parameters is not None:
            validate_data_type(parameters, dict, error_prefix="Input parameters dictionary")
        if parameters is None:
            parameters = {}

        self.thetas = thetas
        self.ps = ps
        self.model = model
        self.window_size = window_size or params.get("window_size") or params.get("window_size")
        self.step_size = step_size or parameters.get("step_size") or params.get("step_size")
        self.train_size = train_size or parameters.get("train_size") or params.get("train_size")
        self.epochs = epochs or parameters.get("epochs") or params.get("epochs")
        self.batch_size = batch_size or parameters.get("batch_size") or params.get("batch_size")
        self.loss = loss or parameters.get("loss") or params.get("loss")
        self.learn_rate = learn_rate or parameters.get("learn_rate") or params.get("learn_rate")

        validate_data_type(self.window_size, int, error_prefix="window size")
        validate_data_type(self.step_size, int, error_prefix="step size")
        validate_data_type(self.train_size, float, error_prefix="training size")
        validate_data_type(self.epochs, int, error_prefix="epochs")
        validate_data_type(self.batch_size, int, error_prefix="batch size")
        validate_data_type(self.loss, str, error_prefix="loss function")
        validate_data_type(self.learn_rate, float, error_prefix="learning rate")

        if self.window_size > PARAMETERS.get("stdm_parameters").get("steps"):
            raise ValueError("Window size cannot be larger than steps")

    def _make_sequences(self):
        thetas_temp = self.thetas[:: self.step_size]
        ps_temp = self.ps[:: self.step_size]

        X, y = [], []

        for k in range(thetas_temp.shape[1]):
            for i in range(thetas_temp.shape[0] - self.window_size):
                X.append(
                    np.stack(
                        (
                            thetas_temp[i: i + self.window_size, k],
                            ps_temp[i: i + self.window_size, k],
                        ),
                        axis=1,
                    )
                )
                y.append(
                    np.stack(
                        [
                            thetas_temp[i + self.window_size, k],
                            ps_temp[i + self.window_size, k],
                        ],
                        axis=-1,
                    )
                )

        return np.array(X), np.array(y)

    def prepare_data(self, shuffle: bool = False):
        self._preprocess_thetas()

        X, y = self._make_sequences()
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=self.train_size)

        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        y_train = torch.Tensor(y_train)
        y_val = torch.Tensor(y_val)

        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train_model(self,
        verbose: bool = False,
        patience: int = None,
    ):
        train_losses = [np.inf]
        validation_losses = [np.inf]

        best_loss = np.inf
        best_model = self.model
        patience_counter = 0
        best_epoch = 0
        hidden = None

        progress_bar = tqdm(total=self.epochs, desc="Training Epochs")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs, hidden)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs, _ = self.model(inputs, hidden)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

            train_loss /= len(self.train_dataset)
            val_loss /= len(self.val_dataset)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model
                best_epoch = epoch
                patience_counter = 0

            train_losses.append(train_loss)
            validation_losses.append(val_loss)

            patience_counter += 1

            if verbose:
                progress_bar.set_postfix(
                    Train_loss=f"{train_loss:.3e}", Val_loss=f"{val_loss:.3e}"
                )
                progress_bar.update(1)

            if patience_counter == patience and epoch < self.epochs:
                print("\nPatience exceeded. Stopping training.")
                break

        progress_bar.close()
        print(f"Best epoch: {best_epoch}")

        self.train_losses = train_losses[1:]
        self.validation_losses = validation_losses[1:]
        self.model = best_model

    def plot_losses(self):
        plt.figure(figsize=(5, 3))
        plt.plot(self.train_losses, color='tab:blue')
        plt.plot(self.validation_losses, color='tab:orange')
        plt.grid(alpha=0.1)
        plt.show()


class Validation(HelperClass):
    """
    A class used to validate a machine learning model on a given dataset.

    Attributes
    ----------
    thetas : numpy.ndarray
        An array of thetas values.
    ps : numpy.ndarray
        An array of ps values.
    model : torch.nn.Module
        A PyTorch model to be validated.
    parameters : dict, optional
        A dictionary of input parameters, by default None.
    window_size : int, optional
        The size of the window for creating sequences, by default None.
    step_size : int, optional
        The step size for creating sequences, by default None.
    train_size : float, optional
        The size of the training set, by default None.
    """
    def __init__(self, thetas, ps, model, parameters: dict = None, window_size: int = None, step_size: int = None, train_size: float = None):
        params = PARAMETERS.get("machine_learning_parameters")

        validate_data_type(params, dict, error_prefix="Parameters dictionary")

        if parameters is not None:
            validate_data_type(parameters, dict, error_prefix="Input parameters dictionary")
        if parameters is None:
            parameters = {}

        self.thetas = thetas
        self.ps = ps
        self.model = model
        self.window_size = window_size or params.get("window_size") or params.get("window_size")
        self.step_size = step_size or parameters.get("step_size") or params.get("step_size")
        self.train_size = train_size or parameters.get("train_size") or params.get("train_size")

        validate_data_type(self.window_size, int, error_prefix="window size")
        validate_data_type(self.step_size, int, error_prefix="step size")
        validate_data_type(self.train_size, float, error_prefix="training size")

    def _make_sequences(self):
        thetas_temp = self.thetas[:: self.step_size]
        ps_temp = self.ps[:: self.step_size]

        X, y = [], []

        for k in range(thetas_temp.shape[1]):
            X.append(
                np.stack(
                    (
                        thetas_temp[: self.window_size, k],
                        ps_temp[: self.window_size, k],
                    ),
                    axis=1,
                )
            )
            y.append(
                np.stack(
                    [
                        thetas_temp[self.window_size:, k],
                        ps_temp[self.window_size:, k],
                    ],
                    axis=-1,
                )
            )

        return np.array(X), np.array(y)

    def prepare_data(self):
        self._preprocess_thetas()

        X, y = self._make_sequences()

        self.X = torch.Tensor(X)
        self.y = y

    def validate_model(self, verbose: bool = False):
        self.model.eval()

        window_size = self.X.shape[1]
        num_predictions = self.y.shape[1]
        final_preds = []

        progress_bar = tqdm(total=self.X.shape[0], desc="Predictions")

        for i in range(self.X.shape[0]):
            hidden = None
            temp_pred = np.copy(self.X[i])

            with torch.no_grad():
                for _ in range(num_predictions):
                    inputs = torch.Tensor(temp_pred[np.newaxis, -window_size:]).to(
                        self.device
                    )
                    pred, hidden = self.model(inputs, hidden)
                    pred = pred.cpu().numpy()
                    temp_pred = np.concatenate((temp_pred, pred), axis=0)

            if verbose:
                progress_bar.set_postfix()
                progress_bar.update(1)

            final_preds.append(temp_pred[window_size:])

        progress_bar.close()

        self.final_preds = np.array(final_preds)

    def get_data(self):
        return np.array(self.X), np.array(self.y)

    def get_predictions(self):
        return self.final_preds

    def plot_2d(self):
        for k in range(self.final_preds.shape[0]):
            plt.figure(figsize=(4, 2))
            plt.plot(self.final_preds[k, :, 0], self.final_preds[k, :, 1], "bo", markersize=1)
            plt.plot(self.y[k, :, 0], self.y[k, :, 1], "ro", markersize=0.5)
            plt.plot(self.final_preds[k, 0, 0], self.final_preds[k, 0, 1], "bo", markersize=5)

            plt.xlim(-0.1, 2.05 * np.pi)
            plt.ylim(-1.5, 1.5)

        plt.show()

    def plot_1d(self):
        num_plots = 3
        steps = 200

        fig, ax = plt.subplots(num_plots, 2, figsize=(10, 2 * num_plots))

        for k in range(num_plots):
            ax[k, 0].set_title(f"Thetas {k+1}")
            ax[k, 0].plot(np.arange(steps), self.y[k, :steps, 0], "tab:blue")
            ax[k, 0].plot(np.arange(steps), self.final_preds[k, :steps, 0], "tab:orange")
            ax[k, 1].set_title(f"Ps {k+1}")
            ax[k, 1].plot(np.arange(steps), self.y[k, :steps, 1], "tab:blue")
            ax[k, 1].plot(np.arange(steps), self.final_preds[k, :steps, 1], "tab:orange")

        plt.tight_layout()
        plt.show()


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        params = PARAMETERS.get("model_parameters")

        self.input_size = 2
        self.hidden_size = params.get("hidden_units")
        self.num_layers = params.get("num_layers")
        self.dropout = params.get("dropout")

        self.RNN = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=self.dropout,
            num_layers=self.num_layers,
        )

        self.dense = torch.nn.Linear(in_features=self.hidden_size, out_features=self.input_size)

    def get_total_number_of_params(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters = {pytorch_total_params}")

    def forward(self, input, hidden):
        out, hidden = self.RNN(input, hidden)
        out = self.dense(out[:, -1, :])
        return out, hidden
