import torch
from torch import nn


class CommNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CommNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense_input_size = self.feature_size()
        self.dense = nn.Sequential(
            nn.Linear(in_features=self.dense_input_size, out_features=128),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.adv = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.output_dim)
        )
        self.val = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, state, batch_size, time_step, hidden_state, cell_state):
        state = state.view(batch_size * time_step, self.dense_input_size)

        dense_out = self.dense(state)  # output [batch_size * time_step, 128]
        lstm_in = dense_out.view(batch_size, time_step, 128)
        lstm_out = self.lstm(lstm_in, (hidden_state, cell_state))  # output[0] [batch_size, time_step, 128]
        out = lstm_out[0][:, time_step - 1, :]  # output [batch_size, 128]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        adv_out = self.adv(out)  # output [128, output_dim]
        val_out = self.val(out)  # output [128, 1]

        q_out = val_out.expand(batch_size, self.output_dim) + (
                adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(batch_size, self.output_dim))
        # output [batch_size, output_dim]
        return q_out, (h_n, c_n)

    def feature_size(self):
        return torch.zeros(self.input_dim, self.input_dim, 5, requires_grad=False).view(1, -1).size(1)
