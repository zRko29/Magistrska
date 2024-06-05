import torch
from src.BaseRNN import BaseRNN
from typing import List


class Model(BaseRNN):
    def __init__(self, **params):
        super(Model, self).__init__()
        self.save_hyperparameters()

        nonlin_hidden = params.get("nonlinearity_hidden")
        self.nonlin_lin = self.configure_non_linearity(params.get("nonlinearity_lin"))

        self.loss = self.configure_loss(params.get("loss"))
        self.accuracy = self.configure_accuracy("path_accuracy", 1.0e-5)

        self.num_rnn_layers: int = params.get("num_rnn_layers")
        self.num_lin_layers: int = params.get("num_lin_layers")
        self.sequence_type: str = params.get("sequence_type")
        dropout: float = params.get("dropout")
        self.lr: float = params.get("lr")
        self.optimizer: str = params.get("optimizer")

        # ----------------------
        # NOTE: This logic is kept so that variable layer sizes can be reimplemented in the future
        rnn_layer_size: int = params.get("hidden_size")
        lin_layer_size: int = params.get("linear_size")

        self.hidden_sizes: List[int] = [rnn_layer_size] * self.num_rnn_layers
        self.linear_sizes: List[int] = [lin_layer_size] * (self.num_lin_layers - 1)
        # ----------------------

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        self.rnns.append(
            torch.nn.RNNCell(2, self.hidden_sizes[0], nonlinearity=nonlin_hidden)
        )
        for layer in range(self.num_rnn_layers - 1):
            self.rnns.append(
                torch.nn.RNNCell(
                    self.hidden_sizes[layer],
                    self.hidden_sizes[layer + 1],
                    nonlinearity=nonlin_hidden,
                )
            )

        # Create the linear layers
        self.lins = torch.nn.ModuleList([])
        if self.num_lin_layers == 1:
            self.lins.append(torch.nn.Linear(self.hidden_sizes[-1], 2))
        elif self.num_lin_layers > 1:
            self.lins.append(
                torch.nn.Linear(self.hidden_sizes[-1], self.linear_sizes[0])
            )
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(
                    torch.nn.Linear(
                        self.linear_sizes[layer], self.linear_sizes[layer + 1]
                    )
                )
            self.lins.append(torch.nn.Linear(self.linear_sizes[-1], 2))
        self.dropout = torch.nn.Dropout(p=dropout)

        # takes care of dtype
        self.to(torch.double)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        # h_ts[i].shape = [batch_size, hidden_sizes[i]]
        h_ts = self._init_hidden(batch_size, self.hidden_sizes)

        outputs = []
        # rnn layers
        for t in range(seq_len):
            h_ts[0] = self.rnns[0](x[t], h_ts[0])
            h_ts[0] = self.dropout(h_ts[0])
            for layer in range(1, self.num_rnn_layers):
                h_ts[layer] = self.rnns[layer](h_ts[layer - 1], h_ts[layer])
                h_ts[layer] = self.dropout(h_ts[layer])
            outputs.append(h_ts[-1])

        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)

        # linear layers
        for layer in range(self.num_lin_layers):
            outputs = self.lins[layer](outputs)
            outputs = self.nonlin_lin(outputs)

        return outputs
