import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error


class Waterlevel_Model(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_length=1):
        super(Waterlevel_Model, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.output_length = output_length

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        _, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1:, :]
        outputs = []

        for t in range(self.output_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            outputs.append(output)

            if self.training and torch.rand(1) < teacher_forcing_ratio and tgt is not None:
                decoder_input = tgt[:, t:t+1, :]
            else:
                decoder_input = output

        return torch.cat(outputs, dim=1)


def RMSE(output, target):
    return np.sqrt(mean_squared_error(output, target))


def MAE(output, target):
    return np.mean(np.abs((output - target)))


metrics = {
    'RMSE': RMSE,
    'MAE': MAE,
}
