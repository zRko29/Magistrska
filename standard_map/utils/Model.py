import torch
import os
import yaml

torch.manual_seed(42)

ROOT_DIR = os.getcwd()
MAIN_DIR = os.path.join(ROOT_DIR, "standard_map")
DATA_DIR = os.path.join(MAIN_DIR, "data")
CONFIG_DIR = os.path.join(MAIN_DIR, "config")

with open(os.path.join(CONFIG_DIR, "parameters.yaml"), "r") as file:
    PARAMETERS = yaml.safe_load(file)


class Model(torch.nn.Module):
    """
    A class for the model.
    """
    def __init__(self, device, hidden_size1: int = None, hidden_size2: int = None, dropout_prob: float = 0.1):
        super(Model, self).__init__()
        params = PARAMETERS.get("model_parameters")

        self.device = device
        self.hidden_size1 = hidden_size1 or params.get("hidden_size1")
        self.hidden_size2 = hidden_size2 or params.get("hidden_size2")
        self.dropout_prob = dropout_prob or params.get("dropout_prob")

        self.rnn1 = torch.nn.RNNCell(2, self.hidden_size1, device=self.device, dtype=torch.double)
        self.dropout1 = torch.nn.Dropout(p=self.dropout_prob)
        self.rnn2 = torch.nn.RNNCell(self.hidden_size1, self.hidden_size2, device=self.device, dtype=torch.double)
        self.dropout2 = torch.nn.Dropout(p=self.dropout_prob)
        self.linear = torch.nn.Linear(self.hidden_size2, 2, device=self.device, dtype=torch.double)

    def _init_hidden(self, shape0: int, shape1: int):
        return torch.zeros(shape0, shape1, device=self.device, dtype=torch.double)

    def forward(self, input, future: int = 0):
        outputs = []
        h_t1 = self._init_hidden(input.shape[0], self.hidden_size1)
        h_t2 = self._init_hidden(input.shape[0], self.hidden_size2)

        for input_t in input.split(1, dim=2):
            input_t = input_t.squeeze(2)
            h_t1 = self.rnn1(input_t, h_t1)
            h_t1 = self.dropout1(h_t1)
            h_t2 = self.rnn2(h_t1, h_t2)
            h_t2 = self.dropout2(h_t2)
            output = self.linear(h_t2)

            outputs.append(output)

        for _ in range(future):  # if we want to do autoregression
            h_t1 = self.rnn1(output, h_t1)
            h_t1 = self.dropout1(h_t1)
            h_t2 = self.rnn2(h_t1, h_t2)
            h_t2 = self.dropout2(h_t2)
            output = self.linear(h_t2)

            outputs.append(output)

        return torch.stack(outputs, dim=2)
