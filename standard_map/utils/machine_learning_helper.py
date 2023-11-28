import numpy as np
from tqdm import tqdm
from utils.general_helper import HelperClass
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class Training(HelperClass):
    """
    A class for training a machine learning model.
    """

    def _make_sequences(self):
        thetas_temp = self.thetas[:: self.step_size]
        ps_temp = self.ps[:: self.step_size]

        X, y = [], []

        for k in range(thetas_temp.shape[1]):
            for i in range(thetas_temp.shape[0] - self.window_size):
                X.append(
                    np.stack(
                        (
                            thetas_temp[i : i + self.window_size, k],
                            ps_temp[i : i + self.window_size, k],
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
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.train_size
        )

        X_train = torch.Tensor(X_train)
        X_val = torch.Tensor(X_val)
        y_train = torch.Tensor(y_train)
        y_val = torch.Tensor(y_val)

        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=shuffle
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    # @torch.compile
    def train_model(
        self,
        verbose: bool = False,
        patience: int = None,
    ):
        train_losses = [np.inf]
        validation_losses = [np.inf]

        best_loss = np.inf
        best_model = self.model
        patience_counter = 0
        best_epoch = 0

        progress_bar = tqdm(total=self.epochs, desc="Training Epochs")

        for epoch in range(self.epochs):
            train_loss = self._train_step()
            val_loss = self._test_step()

            train_losses.append(train_loss)
            validation_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model
                best_epoch = epoch
                patience_counter = 0

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

    def _train_step(self):
        hidden = None
        self.model.train()
        train_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs, _ = self.model(inputs, hidden)
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(self.train_dataset)

        return train_loss

    def _test_step(self):
        hidden = None
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.model(inputs, hidden)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(self.val_dataset)

        return val_loss

    def plot_losses(self):
        plt.figure(figsize=(5, 3))
        plt.plot(self.train_losses, color="tab:blue")
        plt.plot(self.validation_losses, color="tab:orange")
        plt.grid(alpha=0.1)
        plt.show()


class Validation(HelperClass):
    """
    A class used to validate a machine learning model on a given dataset.
    """

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

    # @torch.compile
    def validate_model(self, X=None, verbose: bool = False, model=None):
        if model is not None:
            self.model = model

        if X is not None:
            self.X = X

        window_size = self.X.shape[1]
        num_predictions = self.y.shape[1]
        final_preds = []

        progress_bar = tqdm(total=self.X.shape[0], desc="Predictions")

        self.model.eval()
        for i in range(self.X.shape[0]):
            hidden = None

            with torch.no_grad():
                temp_pred = np.copy(self.X[i])
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

    def compare_predictions(self, model, X_test, y_test):
        test_preds = self.validate_model(model=model, X=X_test)

        return np.mse(test_preds, y_test)

    def plot_2d(self):
        num_subplots = self.final_preds.shape[0]
        plots_per_image = 4

        for k in range(0, num_subplots, plots_per_image):
            plt.figure(figsize=(8, 8))

            for i in range(plots_per_image):
                subplot_index = k + i + 1
                if subplot_index <= num_subplots:
                    plt.subplot(2, 2, i + 1)
                    plt.plot(
                        self.final_preds[subplot_index - 1, :, 0],
                        self.final_preds[subplot_index - 1, :, 1],
                        "bo",
                        markersize=1,
                    )
                    plt.plot(
                        self.y[subplot_index - 1, :, 0],
                        self.y[subplot_index - 1, :, 1],
                        "ro",
                        markersize=0.5,
                    )
                    plt.plot(
                        self.final_preds[subplot_index - 1, 0, 0],
                        self.final_preds[subplot_index - 1, 0, 1],
                        "bo",
                        markersize=5,
                    )

                    plt.xlim(-0.1, 2.05 * np.pi)
                    plt.ylim(-1.5, 1.5)

            plt.tight_layout()
            plt.show()

    def plot_1d(self):
        num_plots = 3
        steps = 200

        fig, ax = plt.subplots(num_plots, 2, figsize=(10, 2 * num_plots))

        for k in range(num_plots):
            ax[k, 0].set_title(f"Thetas {k+1}")
            ax[k, 0].plot(np.arange(steps), self.y[k, :steps, 0], "tab:blue")
            ax[k, 0].plot(
                np.arange(steps), self.final_preds[k, :steps, 0], "tab:orange"
            )
            ax[k, 1].set_title(f"Ps {k+1}")
            ax[k, 1].plot(np.arange(steps), self.y[k, :steps, 1], "tab:blue")
            ax[k, 1].plot(
                np.arange(steps), self.final_preds[k, :steps, 1], "tab:orange"
            )

        plt.tight_layout()
        plt.show()
