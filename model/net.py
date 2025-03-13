import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error


class Waterlevel_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Waterlevel_Model, self).__init__()
        # self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size, output_size)

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        # self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # x = x.permute(0, 2, 1)
        # x = self.conv1d(x)
        # x = self.relu(x)
        # x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        attention_weights = self.attention(out)  # 输出形状: (batch_size, seq_len, 1)
        out = torch.sum(out * attention_weights, dim=1)
        out = self.fc(out)
        # out = self.fc1(x)
        # out = self.transformer(out)
        # out = self.fc2(out)
        # print(out)
        return out


def RMSE(output, target):
    return np.sqrt(mean_squared_error(output, target))


metrics = {
    'RMSE': RMSE,
}
