import torch
from torch import nn


# Used in grid world
class DuelingRCNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DuelingRCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # output dim = input dim = 11

        self.lstm_input_size = self.feature_size()
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=1, batch_first=True)
        self.adv = nn.Linear(in_features=128, out_features=self.output_dim)
        self.val = nn.Linear(in_features=128, out_features=1)

    def forward(self, state, batch_size, time_step, hidden_state, cell_state):
        state = state.view(batch_size * time_step, 1, self.input_dim, self.input_dim)

        conv_out = self.conv(state)
        lstm_in = conv_out.view(batch_size, time_step, self.lstm_input_size)
        lstm_out = self.lstm(lstm_in, (hidden_state, cell_state))
        out = lstm_out[0][:, time_step - 1, :]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        adv_out = self.adv(out)  # output [128, output_dim]
        val_out = self.val(out)  # output [128, 1]

        q_out = val_out.expand(batch_size, self.output_dim) + (
                adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(batch_size, self.output_dim))
        # output [batch_size, output_dim]
        return q_out, (h_n, c_n)

    def feature_size(self):
        return self.conv(torch.zeros(1, 1, self.input_dim, self.input_dim, requires_grad=True)).view(1, -1).size(1)


class PreprocessingDRCNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PreprocessingDRCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input 11x11x5
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # output dim = input dim = 11

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
        # (batch_size, C_in, height, width)
        state = state.view(batch_size * time_step, 5, self.input_dim, self.input_dim)

        # conv_out = self.conv(state)  # output [batch_size * time_step, 64, 11, 11]
        # dense_in = self.conv(state).view(batch_size * time_step, self.dense_input_size)
        # dense_out = self.dense(self.conv(state).view(batch_size * time_step, self.dense_input_size))  # output [batch_size * time_step, 128]
        # lstm_in = self.dense(self.conv(state).view(batch_size * time_step, self.dense_input_size)).view(batch_size, time_step, 128)
        lstm_out = self.lstm(self.dense(self.conv(state).view(batch_size * time_step, self.dense_input_size)).view(batch_size, time_step, 128),
                             (hidden_state, cell_state))  # output[0] [batch_size, time_step, 128]
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
        return self.conv(torch.zeros(1, 5, self.input_dim, self.input_dim, requires_grad=False)).view(1, -1).size(1)


class CommDRCNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CommDRCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input 11x11x5
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # output dim = input dim = 11

        self.dense_input_size = self.feature_size()
        self.dense_conv_out = nn.Sequential(
            nn.Linear(in_features=self.dense_input_size, out_features=128),
            nn.ReLU()
        )
        self.dense_message = nn.Sequential(
            nn.Linear(in_features=18, out_features=128),
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

    def forward(self, state, teammate_message, batch_size, time_step, hidden_state, cell_state):
        # (batch_size, C_in, height, width)
        state = state.view(batch_size * time_step, 5, self.input_dim, self.input_dim)
        # message will be sent to teammate at this time_step
        # (batch_size, radio_num_words * num_classes)
        teammate_message = teammate_message.view(batch_size * time_step, 18)
        # (batch_size, number of q_values)
        # message = message.view(batch_size * time_step, )

        # conv_out = self.conv(state)  # output [batch_size * time_step, 64, 11, 11]
        # dense_in = self.conv(state).view(batch_size * time_step, self.dense_input_size)
        # conv_dense_out = self.dense_conv_out(self.conv(state).view(batch_size * time_step, self.dense_input_size))  # output [batch_size * time_step, 128]
        # message_dense_out = self.dense_message(teammate_message)  # output [batch_size * time_step, 128]
        lstm_in = self.dense_conv_out(self.conv(state).view(batch_size * time_step, self.dense_input_size)) + self.dense_message(teammate_message)
        # lstm_in = lstm_in.view(batch_size, time_step, 128)
        lstm_out = self.lstm(lstm_in.view(batch_size, time_step, 128), (hidden_state, cell_state))  # output[0] [batch_size, time_step, 128]
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
        return self.conv(torch.zeros(1, 5, self.input_dim, self.input_dim, requires_grad=False)).view(1, -1).size(1)
