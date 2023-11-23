import numpy as np
import torch
import torch.optim as optim


class HelperClass:

    def set_criterion(self, loss: str = None):
        """
        Sets the loss criterion for the model.

        Args:
            loss (str): The name of the loss function to use. If None or "mse", the mean squared error loss will be used. If "huber", the Huber loss will be used.
        """
        if loss is None or loss == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss == "huber":
            self.criterion = torch.nn.HuberLoss()
        else:
            raise ValueError("Invalid loss function. Valid options are 'mse' and 'huber'.")

    def set_optimizer(self, optimizer: str = None):
        """
        Sets the optimizer for the model based on the given optimizer string.

        Args:
            optimizer (str): The optimizer to use. Valid options are "adam", "adamw", "radam", "adamax", "nadam", and "rprop". If None, defaults to "adam".
        """
        if optimizer is None or optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)
        elif optimizer == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learn_rate)
        elif optimizer == "radam":
            self.optimizer = optim.RAdam(self.model.parameters(), lr=self.learn_rate)
        elif optimizer == "adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.learn_rate)
        elif optimizer == "nadam":
            self.optimizer = optim.NAdam(self.model.parameters(), lr=self.learn_rate)
        elif optimizer == "rprop":
            self.optimizer = optim.Rprop(self.model.parameters(), lr=self.learn_rate)
        else:
            raise ValueError("Invalid optimizer. Valid options are 'adam', 'adamw', 'radam', 'adamax', 'nadam', and 'rprop'.")

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _preprocess_thetas(self):
        return self.thetas / np.pi - 1

    def _postprocess_thetas(self):
        return (self.thetas + 1) * np.pi
