import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.general_helper import Miscellaneous

import torch
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)


class ModelTrainer(Miscellaneous):
    """
    A class for training a machine learning model.
    """

    def __init__(self, device):
        super(ModelTrainer, self).__init__()
        self.device = device

    def prepare_data(self, thetas: np.ndarray, ps: np.ndarray, shuffle: bool = None):
        self.shuffle = shuffle or self.shuffle
        thetas = self.preprocess_thetas(thetas)

        # data.shape = [init_points, 2, steps]
        data = np.stack([thetas.T, ps.T], axis=1)
        t = int(len(data) * self.train_size)

        if self.shuffle:
            np.random.shuffle(data)

        train_inputs = torch.from_numpy(data[:t, :, :-1])
        train_outputs = torch.from_numpy(data[:t, :, 1:])
        val_inputs = torch.from_numpy(data[t:, :, :-1])
        val_outputs = torch.from_numpy(data[t:, :, 1:])

        # for easier access to batches
        train_tensor = TensorDataset(train_inputs, train_outputs)
        self.train_loader = DataLoader(train_tensor, batch_size=self.batch_size)

        val_tensor = TensorDataset(val_inputs, val_outputs)
        self.val_loader = DataLoader(val_tensor, batch_size=self.batch_size)

    # @torch.compile
    def train_model(
        self,
        verbose: bool = False,
        epochs: int = None,
    ):
        self.epochs = epochs or self.epochs
        train_losses = [np.inf]
        val_losses = [np.inf]

        best_loss = np.inf
        best_model = self.model
        best_epoch = 0

        progress_bar = tqdm(total=self.epochs, desc="Training Epochs")

        for epoch in range(self.epochs):
            train_loss = self._train_step()
            val_loss = self._test_step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = self.model
                best_epoch = epoch

            if verbose:
                progress_bar.set_postfix(
                    Train_loss=f"{train_loss:.5f}", Val_loss=f"{val_loss:.5f}"
                )
                progress_bar.update(1)

        progress_bar.close()
        print(f"Best epoch: {best_epoch}")

        self.train_losses = train_losses[1:]
        self.validation_losses = val_losses[1:]
        self.model = best_model

    def _train_step(self):
        train_loss = 0.0
        self.model.train()
        for inputs, outputs in self.train_loader:
            inputs = inputs.to(self.device)
            outputs = outputs.to(self.device)

            predicted = self.model(inputs)
            loss = self.criterion(predicted, outputs)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(self.train_loader.dataset)

        return train_loss

    def _test_step(self):
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for inputs, outputs in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)

                predicted = self.model(inputs)
                loss = self.criterion(predicted, outputs)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(self.val_loader.dataset)

        return val_loss

    def plot_losses(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.train_losses, color="tab:blue", label="Training loss")
        plt.plot(self.validation_losses, color="tab:orange", label="Validation loss")
        plt.grid(alpha=0.1)
        plt.legend()
        plt.show()

    def do_autoregression(self, thetas: np.ndarray, ps: np.ndarray, regression_seed: int = 1):
        thetas = self.preprocess_thetas(thetas)

        # data.shape = [init_points, 2, steps]
        data = np.stack([thetas.T, ps.T], axis=1)

        assert data.shape[2] > regression_seed, "regression_seed must be smaller than number of steps"

        inputs = torch.from_numpy(data[:, :, :regression_seed])
        self.outputs = torch.from_numpy(data[:, :, regression_seed:])

        with torch.inference_mode():
            inputs = inputs.to(self.device)
            self.outputs = self.outputs.to(self.device)

            self.predicted = self.model(inputs, future=data.shape[2] - regression_seed)
            self.predicted = self.predicted[:, :, regression_seed:]  # remove regression_seed, because it wasn't predicted
            loss = self.criterion(self.predicted, self.outputs)

        print(f"Loss: {loss.item():.5f}")

    def plot_2d(self):
        self.predicted = self.predicted.cpu().numpy()
        self.outputs = self.outputs.cpu().numpy()

        plt.figure(figsize=(10, 7))
        for k in range(self.predicted.shape[0]):
            plt.plot(self.predicted[k, 0], self.predicted[k, 1], "bo", markersize=1)
            plt.plot(self.outputs[k, 0], self.outputs[k, 1], "ro", markersize=0.5)
        plt.grid(alpha=0.1)
        plt.show()
