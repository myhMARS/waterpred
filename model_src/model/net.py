import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error


class Waterlevel_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Waterlevel_Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out[:, -1, :]


def RMSE(output, target):
    return np.sqrt(mean_squared_error(output, target))


def MAE(output, target):
    return np.mean(np.abs((output - target)))


metrics = {
    'RMSE': RMSE,
    'MAE': MAE,
}
