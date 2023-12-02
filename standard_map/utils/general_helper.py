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
        learn_rate: float = None,
    ):
        params = PARAMETERS.get("machine_learning_parameters")

        self.train_size = train_size or params.get("train_size")
        self.epochs = epochs or params.get("epochs")
        self.batch_size = batch_size or params.get("batch_size")
        self.loss = loss or params.get("loss")
        self.learn_rate = learn_rate or params.get("learn_rate")

    def set_model(self, model=None):
        model.double()
        self.model = model
        # self.model = torch.compile(model)

    def set_criterion(self, loss: str = None):
        if loss is None or loss == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss == "huber":
            self.criterion = torch.nn.HuberLoss()
        else:
            raise ValueError(
                "Invalid loss function. Valid options are 'mse' and 'huber'."
            )

    def set_optimizer(self, optimizer: str = None):
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
            raise ValueError(
                "Invalid optimizer. Valid options are 'adam', 'adamw', 'radam', 'adamax', 'nadam', and 'rprop'."
            )

    def set_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_thetas(self, thetas: np.ndarray):
        return thetas / np.pi - 1
