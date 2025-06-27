import torch
import numpy as np
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error

from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class WaterLevelGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=1):
        super(WaterLevelGCN, self).__init__()

        # Transformer：每个节点的时间序列特征 [window] 做编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # GCN：每个节点的 transformer 表征作为输入，建模空间关系
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        x: [num_nodes, window] —— 每个节点的时间序列特征
        edge_index: 图结构
        """
        x = self.transformer(x)  # [N, window]
        x = x.squeeze(-1)
        # GCN 编码空间关系
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.gcn3(x, edge_index)  # 输出 [N, output_dim]

        return x


def RMSE(output, target):
    return np.sqrt(mean_squared_error(output, target))


def MAE(output, target):
    return np.mean(np.abs((output - target)))


metrics = {
    'RMSE': RMSE,
    'MAE': MAE,
}
