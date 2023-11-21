import torch
import os
import yaml

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

ROOT_DIR = os.getcwd()
MAIN_DIR = os.path.join(ROOT_DIR, "standard_map")
DATA_DIR = os.path.join(MAIN_DIR, "data")
CONFIG_DIR = os.path.join(MAIN_DIR, "config")

with open(os.path.join(CONFIG_DIR, "parameters.yaml"), "r") as file:
    PARAMETERS = yaml.safe_load(file)

class Model(torch.nn.Module):
    def __init__(self, parameters: dict = None):
        super().__init__()

        params = PARAMETERS.get("model_parameters")

        if parameters is None:
            parameters = {}

        self.input_size = 2
        self.hidden_size = parameters.get("hidden_units") or params.get("hidden_units")
        self.num_layers = parameters.get("num_layers") or params.get("num_layers")
        self.dropout = parameters.get("dropout") or params.get("dropout")

        self.RNN = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=self.dropout,
            num_layers=self.num_layers,
        )

        self.dense = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.input_size, bias=True
        )

    def get_total_number_of_params(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters = {pytorch_total_params}")

    def forward(self, input, hidden):
        out, hidden = self.RNN(input, hidden)
        out = self.dense(out[:, -1, :])
        return out, hidden
