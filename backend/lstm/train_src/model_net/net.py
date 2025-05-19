import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error


class Waterlevel_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Waterlevel_Model, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1]
        out = self.output_proj(x)
        return out


def RMSE(output, target):
    return np.sqrt(mean_squared_error(output, target))


def MAE(output, target):
    return np.mean(np.abs((output - target)))


metrics = {
    'RMSE': RMSE,
    'MAE': MAE,
}
