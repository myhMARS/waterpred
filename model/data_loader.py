import torch
import pandas as pd
from torch.utils.data import Dataset


class WaterLevelDataset(Dataset):
    def __init__(self, file_path, train, seq_length, pred_length, scaler, split_ratio=0.8):
        self.file_path = file_path
        self.train = train
        self.df = pd.read_csv(self.file_path)

        X = scaler[0].fit_transform(self.df.drop(columns=['waterlevels']))
        y = scaler[1].fit_transform(self.df.waterlevels_smooth.values.reshape(-1, 1))
        sequences = []
        for i in range(len(X) - seq_length - pred_length):
            seq_x = (X[i: i + seq_length, :])
            seq_y = (y[i + seq_length: i + seq_length + pred_length, -1])
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
        return self.df.drop(columns=['waterlevels']), self.df.waterlevels_smooth
