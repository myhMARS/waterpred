import torch
import pandas as pd
from torch.utils.data import Dataset


class WaterLevelDataset(Dataset):
    def __init__(self, X_list, y_list, train, seq_length, pred_length, scaler, split_ratio=0.8):
        self.train = train
        self.X_list = X_list
        self.y_list = y_list

        datas_X = pd.concat(X_list, axis=0)
        datas_y = pd.concat(y_list, axis=0)
        scaler[0].fit(datas_X)
        scaler[1].fit(datas_y.values.reshape(-1, 1))

        sequences = []
        for X_data, y_data in zip(X_list, y_list):
            X = scaler[0].transform(X_data)
            y = scaler[1].transform(y_data.values.reshape(-1, 1))

            for i in range(len(X) - seq_length - pred_length):
                seq_x = X[i: i + seq_length, :]
                seq_y = y[i + seq_length: i + seq_length + pred_length, 0]
                sequences.append((seq_x, seq_y))
        train_size = int(len(sequences) * split_ratio)  # 70% 训练
        data_train, data_test = sequences[:train_size], sequences[train_size:]
        self.train_data = data_train
        self.test_data = data_test

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            seq_x, seq_y = self.train_data[idx]
        else:
            seq_x, seq_y = self.test_data[idx]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def get_source(self):
        return self.X_list, self.y_list
