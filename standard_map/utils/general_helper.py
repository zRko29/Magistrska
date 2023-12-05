import torch
import torch.optim as optim
import os
import yaml
import numpy as np

ROOT_DIR = os.getcwd()
MAIN_DIR = os.path.join(ROOT_DIR, "standard_map")
DATA_DIR = os.path.join(MAIN_DIR, "data")
CONFIG_DIR = os.path.join(MAIN_DIR, "config")

with open(os.path.join(CONFIG_DIR, "parameters.yaml"), "r") as file:
    PARAMETERS = yaml.safe_load(file)


class Miscellaneous:
    """
    A class for miscellaneous methods.
    """

    def __init__(
        self,
        train_size: float = None,
        epochs: int = None,
        batch_size: int = None,
        loss: str = None,
        optimizer: str = None,
        learn_rate: float = None,
        shuffle: bool = None,
    ):
        params = PARAMETERS.get("machine_learning_parameters")

        self.train_size = train_size or params.get("train_size")
        self.epochs = epochs or params.get("epochs")
        self.batch_size = batch_size or params.get("batch_size")
        self.loss = loss or params.get("loss")
        self.optimizer = optimizer or params.get("optimizer")
        self.learn_rate = learn_rate or params.get("learn_rate")
        self.shuffle = shuffle or params.get("shuffle")

    def set_model(self, model=None):
        model.double()
        model = model.to(self.device)
        self.model = model
        # self.model = torch.compile(model)

    def set_criterion(self, loss: str = None):
        self.loss = loss or self.loss

        if self.loss is None or self.loss == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.loss == "huber":
            self.criterion = torch.nn.HuberLoss()
        else:
            raise ValueError(
                "Invalid loss function. Valid options are 'mse' and 'huber'."
            )

    def set_optimizer(self, optimizer: str = None):
        self.optimizer = optimizer or self.optimizer

        if self.optimizer is None or self.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)
        elif self.optimizer == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learn_rate)
        elif self.optimizer == "radam":
            self.optimizer = optim.RAdam(self.model.parameters(), lr=self.learn_rate)
        elif self.optimizer == "adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=self.learn_rate)
        elif self.optimizer == "nadam":
            self.optimizer = optim.NAdam(self.model.parameters(), lr=self.learn_rate)
        elif self.optimizer == "rprop":
            self.optimizer = optim.Rprop(self.model.parameters(), lr=self.learn_rate)
        else:
            raise ValueError(
                "Invalid optimizer. Valid options are 'adam', 'adamw', 'radam', 'adamax', 'nadam', and 'rprop'."
            )

    def preprocess_thetas(self, thetas: np.ndarray):
        return thetas / np.pi - 1
