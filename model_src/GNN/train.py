import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler
import random

from model.net import WaterLevelGCN

# 设置字体为微软雅黑，支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def moving_average(data, window=5):
    return data.rolling(window=window).mean()


def create_graph_samples(water_data, edge_index, window=12, horizon=1):
    """
    将时间序列构造成多个图样本。
    water_data: [T, N]，时间步 x 站点数
    edge_index: 图结构
    返回: List[Data]，每个是一个图
    """
    T, N = water_data.shape
    samples = []

    for t in range(window, T - horizon):
        x = water_data[t - window:t]  # [window, N]
        y = water_data[t + horizon]  # [N]

        # 转为 [N, window] 的每个节点特征
        x_feat = x.T  # [N, window]
        y_target = y.reshape(-1, 1)  # [N, 1]

        data = Data(
            x=torch.tensor(x_feat, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y_target, dtype=torch.float)
        )
        samples.append(data)

    return samples


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df_smooth = df[['waterlevels63000120', 'waterlevels63000100', 'waterlevels']].apply(moving_average, window=3)
    data = df_smooth.dropna()
    print(data)
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long).to(device)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    graph_list = create_graph_samples(data_scaled, edge_index)
    random.shuffle(graph_list)
    train_ratio = 0.8
    train_len = int(len(graph_list) * train_ratio)

    train_graphs = graph_list[:train_len]
    eval_graphs = graph_list[train_len:]

    model = WaterLevelGCN(input_dim=12, hidden_dim=64, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_graphs, batch_size=32)
    eval_loader = DataLoader(eval_graphs, batch_size=32)

    train_losses = []
    eval_losses = []

    for epoch in range(150):
        # === Train ===
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)
            loss = F.mse_loss(out, batch.y)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss
        train_losses.append(avg_train_loss)

        # === Eval ===
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for eval_batch in eval_loader:
                eval_batch = eval_batch.to(device)
                out = model(eval_batch.x, eval_batch.edge_index)
                loss = F.mse_loss(out, eval_batch.y)
                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss
        eval_losses.append(avg_eval_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

    # === Plot ===
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(eval_losses, label='Eval Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index).cpu().numpy()  # [N, 1]
            true = batch.y.cpu().numpy()  # [N, 1]

            all_preds.append(pred)
            all_trues.append(true)

    all_preds = np.vstack(all_preds)  # [total_nodes, 1]
    all_trues = np.vstack(all_trues)  # [total_nodes, 1]

    station_num = all_preds.shape[0] // len(graph_list[0].x)  # 每个时间点 N 个站
    N = len(graph_list[0].x)  # 节点数（站点数）
    # 假设你预测的是所有 N 个站点的水位
    # all_preds, all_trues: [T*N, 1]

    T = all_preds.shape[0] // N  # 样本数量
    preds_matrix = all_preds.reshape(T, N)  # [T, N]
    trues_matrix = all_trues.reshape(T, N)  # [T, N]

    # 逆归一化
    min_vals = scaler.data_min_
    max_vals = scaler.data_max_

    preds_inverse = preds_matrix * (max_vals - min_vals) + min_vals
    trues_inverse = trues_matrix * (max_vals - min_vals) + min_vals

    plt.figure(figsize=(15, 4))
    for i in range(N):
        plt.subplot(1, N, i + 1)
        plt.plot(trues_inverse[:50, i], label="True", marker='o')
        plt.plot(preds_inverse[:50, i], label="Pred", marker='x')
        plt.title(f"站点{i}")
        plt.xlabel("时间片")
        plt.ylabel("水位")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()
