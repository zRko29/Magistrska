import torch
import torch.nn as nn
import torch.nn.functional as F
from src.BaseRNN import BaseRNN


class Model(BaseRNN):
    def __init__(self, **params):
        super(Model, self).__init__(**params)

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        self.rnns.append(MinimalGatedCell(2, self.hidden_sizes[0]))
        for layer in range(self.num_rnn_layers - 1):
            self.rnns.append(
                MinimalGatedCell(self.hidden_sizes[layer], self.hidden_sizes[layer + 1])
            )

        # Create the linear layers
        self.lins = torch.nn.ModuleList([])
        self.bn = torch.nn.ModuleList([])

        if self.num_lin_layers == 1:
            self.lins.append(torch.nn.Linear(self.hidden_sizes[-1], 2))
        elif self.num_lin_layers > 1:
            self.lins.append(
                torch.nn.Linear(self.hidden_sizes[-1], self.linear_sizes[0])
            )
            self.bn.append(torch.nn.BatchNorm1d(self.linear_sizes[0]))
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(
                    torch.nn.Linear(
                        self.linear_sizes[layer], self.linear_sizes[layer + 1]
                    )
                )
                self.bn.append(torch.nn.BatchNorm1d(self.linear_sizes[layer + 1]))
            self.lins.append(torch.nn.Linear(self.linear_sizes[-1], 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        # h_ts[i].shape = [batch_size, hidden_sizes[i]]
        h_ts = self._init_hidden(batch_size, self.hidden_sizes)

        outputs = []
        # rnn layers
        for t in range(seq_len):
            h_ts[0] = self.rnns[0](x[t], h_ts[0])
            for layer in range(1, self.num_rnn_layers):
                h_ts[layer] = self.rnns[layer](h_ts[layer - 1], h_ts[layer])
            outputs.append(h_ts[-1])

        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)

        # linear layers
        for layer in range(self.num_lin_layers - 1):
            outputs = self.lins[layer](outputs)
            outputs = outputs.transpose(1, 2)
            outputs = self.bn[layer](outputs)
            outputs = outputs.transpose(1, 2)
            outputs = self.nonlin_lin(outputs)

        outputs = self.lins[-1](outputs)
        outputs = self.nonlin_lin(outputs)

        return outputs


class MinimalGatedCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinimalGatedCell, self).__init__()

        # Parameters for forget gate
        self.weight_fx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_fh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_f = nn.Parameter(torch.Tensor(hidden_size))

        # Parameters for candidate activation
        self.weight_hx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_fx)
        nn.init.kaiming_uniform_(self.weight_fh)
        nn.init.zeros_(self.bias_f)

        nn.init.kaiming_uniform_(self.weight_hx)
        nn.init.kaiming_uniform_(self.weight_hf)
        nn.init.zeros_(self.bias_h)

    def forward(self, h1, h2):
        # Compute forget gate
        f_t = torch.sigmoid(
            F.linear(h1, self.weight_fx, self.bias_f) + F.linear(h2, self.weight_fh)
        )

        # Compute candidate activation
        h_hat_t = torch.tanh(
            F.linear(h1, self.weight_hx, self.bias_h)
            + F.linear(f_t * h2, self.weight_hf)
        )

        # Compute output
        h_t = (1 - f_t) * h2 + f_t * h_hat_t

        return h_t
