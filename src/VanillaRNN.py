import torch
from src.BaseRNN import BaseRNN


class Model(BaseRNN):
    def __init__(self, **params):
        super(Model, self).__init__(**params)

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        for layer in range(self.num_rnn_layers):
            self.rnns.append(
                torch.nn.RNNCell(
                    2,
                    self.hidden_sizes[layer],
                    nonlinearity=self.nonlin_hidden,
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
        self.dropout = torch.nn.Dropout(p=self.dropout)

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
            for layer in range(self.num_rnn_layers):
                h_ts[layer] = self.rnns[layer](x[t], h_ts[layer])
                h_ts[layer] = self.dropout(h_ts[layer])
            outputs.append(h_ts[-1])

        outputs = torch.stack(outputs)
        outputs = outputs.transpose(0, 1)

        # linear layers
        for layer in range(self.num_lin_layers):
            outputs = self.lins[layer](outputs)
            outputs = self.nonlin_lin(outputs)

        return outputs
