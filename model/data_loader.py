import logging

import torch
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils import get_xgboost_feature


class WaterLevelDataset(Dataset):
    def __init__(self, file_path, train, seq_length, pred_length, scaler, split_ratio=0.8):
        self.file_path = file_path
        self.data = []

        df = pd.read_csv(self.file_path)
        # feature_cols = ['precipitation', 'water_level'] + \
        #                [c for c in df.columns if c.startswith('season_')] + \
        #                ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'] + \
        #                [c for c in df.columns if 'precip_avg' in c or 'level_std' in c]

        X = scaler[0].fit_transform(df.drop(columns=['waterlevels']))
        y = scaler[1].fit_transform(df.waterlevels_smooth.values.reshape(-1, 1))
        # xgb_model = XGBRegressor()
        # xgb_model.fit(X[:, :-1], y)
        # xgb_pred = xgb_model.predict(X[:, :-1])
        # logging.info(f'XGBoost RMSE:{np.sqrt(mean_squared_error(xgb_pred, y)):.3f}')
        #
        # X[:, -1] = xgb_pred
        # X_data, y_data = [], []
        sequences = []
        for i in range(len(X) - seq_length - pred_length + 1):
            seq_x = (X[i: i + seq_length, :])
            seq_y = (y[i + seq_length: i + seq_length + pred_length, -1])
            sequences.append((seq_x, seq_y))

        # features = get_xgboost_feature(*train_data)
        data_train, data_test = train_test_split(sequences, train_size=split_ratio)

        self.data = data_train if train else data_test

        # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_x, seq_y = self.data[idx]
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
