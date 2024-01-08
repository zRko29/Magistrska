import torch
import torch.optim as optim


class Miscellaneous:
    """
    A class for miscellaneous methods.
    """

    def set_criterion(self, loss: str):
        if loss == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss == "mae":
            self.criterion = torch.nn.L1Loss()
        elif loss == "huber":
            self.criterion = torch.nn.HuberLoss()

    def set_optimizer(self, optimizer: str, learn_rate: float):
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learn_rate)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learn_rate, momentum=0.9, nesterov=True
            )
