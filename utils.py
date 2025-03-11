import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def show_action_data(model, dataloader, scaler=None):
    model.eval()

    y_pred, y_true = [], []
    for data_batch, labels_batch in dataloader:
        data_batch = data_batch.to(device)

        output_batch = model(data_batch)
        output_batch = output_batch.data.cpu().numpy()
        if scaler:
            output_batch = scaler.inverse_transform(np.array(output_batch).reshape(-1, output_batch.shape[-1]))
            labels_batch = scaler.inverse_transform(np.array(labels_batch).reshape(-1, output_batch.shape[-1]))
        y_pred.extend(output_batch)
        y_true.extend(labels_batch)

    plt.ion()

    fig, ax = plt.subplots()
    ymin = min(np.min(y_pred), np.min(y_true))
    ymax = max(np.max(y_pred), np.max(y_true))
    time_length = 24*7
    y_pred_show, y_true_show = [], []
    for i in range(len(y_pred)):
        y_true_show.append(y_true[i][0])

        if i % len(y_pred[i]) == 0:
            y_pred_show.extend(y_pred[i])

        # if len(y_pred_show) > len(y_pred[i]) - 1:
        #     for _ in range(len(y_pred[i]) - 1):
        #         y_pred_show.pop()
        # y_pred_show.extend(y_pred[i])

        if len(y_pred_show) >= time_length:
            for _ in range(len(y_pred_show) - time_length):
                del y_pred_show[0]
                del y_true_show[0]
        # print(len(y_pred_show), len(y_true_show))
        ax.clear()
        ax.set_ylim([ymin - 0.5, ymax])
        ax.plot(range(len(y_pred_show)), y_pred_show, label=f'prediction {i}', color='r')
        ax.plot(range(len(y_true_show)), y_true_show, label=f'true label {i}', color='b')
        ax.plot(range(len(y_true_show)),
                [i - j - 0.5 for i, j in zip(y_pred_show[:len(y_true_show)], y_true_show)],
                label=f'diff - 0.5 {i}',
                color='g')
        ax.legend()
        plt.draw()
        plt.pause(0.2)


def get_xgboost_feature(X_train, y_train):
    xgboost = XGBRegressor()
    xgboost.fit(X_train, y_train)
    features = xgboost.predict(X_train)
    return features
