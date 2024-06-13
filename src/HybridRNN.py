import torch
from src.BaseRNN import BaseRNN


class Model(BaseRNN):
    def __init__(self, **params):
        super(Model, self).__init__(**params)

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        self.rnns.append(
            torch.nn.RNNCell(2, self.hidden_sizes[0], nonlinearity=self.nonlin_hidden)
        )
        for layer in range(1, self.num_rnn_layers):
            self.rnns.append(
                HybridRNNCell(
                    2,
                    self.hidden_sizes[layer],
                    self.hidden_sizes[layer - 1],
                    nonlinearity=self.nonlin_hidden,
                )
            )

        # Create the linear layers
        self.lins = torch.nn.ModuleList([])
        self.bn_lins = torch.nn.ModuleList([])

        if self.num_lin_layers == 1:
            self.lins.append(torch.nn.Linear(self.hidden_sizes[-1], 2))
        elif self.num_lin_layers > 1:
            self.lins.append(
                torch.nn.Linear(self.hidden_sizes[-1], self.linear_sizes[0])
            )
            self.bn_lins.append(torch.nn.BatchNorm1d(self.linear_sizes[0]))
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(
                    torch.nn.Linear(
                        self.linear_sizes[layer], self.linear_sizes[layer + 1]
                    )
                )
                self.bn_lins.append(torch.nn.BatchNorm1d(self.linear_sizes[layer + 1]))
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
                h_ts[layer] = self.rnns[layer](x[t], h_ts[layer], h_ts[layer - 1])
            outputs.append(h_ts[-1])

        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)

        # linear layers
        for layer in range(self.num_lin_layers - 1):
            outputs = self.lins[layer](outputs)
            outputs = outputs.transpose(1, 2)
            outputs = self.bn_lins[layer](outputs)
            outputs = outputs.transpose(1, 2)
            outputs = self.nonlin_lin(outputs)
        outputs = self.lins[-1](outputs)

        return outputs


class HybridRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, nonlinearity="relu"):
        super(HybridRNNCell, self).__init__()

        self.weight_ih = torch.nn.Parameter(torch.Tensor(hidden_size1, input_size))
        self.bias_ih = torch.nn.Parameter(torch.Tensor(hidden_size1))

        # hidden state at time t-1
        self.weight_hh1 = torch.nn.Parameter(torch.Tensor(hidden_size1, hidden_size1))
        self.bias_hh1 = torch.nn.Parameter(torch.Tensor(hidden_size1))

        # hidden state at previous layer
        self.weight_hh2 = torch.nn.Parameter(torch.Tensor(hidden_size1, hidden_size2))
        self.bias_hh2 = torch.nn.Parameter(torch.Tensor(hidden_size1))

        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "relu":
            self.nonlinearity = torch.nn.functional.relu

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_ih)
        torch.nn.init.zeros_(self.bias_ih)

        torch.nn.init.kaiming_uniform_(self.weight_hh1)
        torch.nn.init.zeros_(self.bias_hh1)

        torch.nn.init.kaiming_uniform_(self.weight_hh2)
        torch.nn.init.zeros_(self.bias_hh2)

    # @torch.jit.script
    def forward(self, input, hidden1, hidden2):
        h_t = self.nonlinearity(
            torch.nn.functional.linear(input, self.weight_ih, self.bias_ih)
            + torch.nn.functional.linear(hidden1, self.weight_hh1, self.bias_hh1)
            + torch.nn.functional.linear(hidden2, self.weight_hh2, self.bias_hh2)
        )
        return h_t
