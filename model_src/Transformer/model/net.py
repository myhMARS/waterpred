import torch
import numpy as np
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error


class Waterlevel_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, max_len=500):
        super(Waterlevel_Model, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('positional_encoding', pe)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)

        x = x + self.positional_encoding[:, :seq_len, :]

        x = self.transformer(x)

        # 取最后一个时间步的表示
        x = x[:, -1, :]

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
