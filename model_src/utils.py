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


def show_action_data(model, dataloader, scaler, res_file=None):
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

    fig, ax = plt.subplots(2, 1)
    ymin = min(np.min(y_pred), np.min(y_true))
    ymax = max(np.max(y_pred), np.max(y_true))
    y_pred_show, y_true_show = [], []
    diff = []
    for i in range(len(y_pred)):
        if len(y_true_show) > len(y_true[i]) - 1:
            for _ in range(len(y_true[i]) - 1):
                y_true_show.pop()
        y_true_show.extend(y_true[i])

        # if i % len(y_pred[i]) == 0:
        #     y_pred_show.extend(y_pred[i])

        if len(y_pred_show) > len(y_pred[i]) - 1:
            for _ in range(len(y_pred[i]) - 1):
                y_pred_show.pop()
        y_pred_show.extend(y_pred[i])
        diff.append(np.mean(np.abs(y_pred[i] - y_true[i])))
        # ax[0].clear()
        # ax[1].clear()
        # if time_length:
        #     if len(y_pred_show) >= time_length:
        #         for _ in range(len(y_pred_show) - time_length):
        #             del y_pred_show[0]
        #             del y_true_show[0]
        #     for x in range(i % len(y_pred[i]), len(y_true_show), len(y_pred[i])):
        # ax[1].axvline(x=len(diff) - len(y_pred[i]) - 1, color='gray', linestyle='--', alpha=0.5)
        # ax[0].axvline(x=len(y_pred_show) - len(y_pred[i]) - 1, color='gray', linestyle='--', alpha=0.5)
        # else:
        #     for x in range(0, len(y_true_show), len(y_pred[i])):
        #         ax[1].axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        #         ax[0].axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        # print(len(y_pred_show), len(y_true_show))

    ax[0].set_ylim([ymin, ymax])
    ax[0].plot(range(len(y_pred_show)), [84.66] * len(y_pred_show), linestyle='--', label='waring levels')
    ax[0].plot(range(len(y_pred_show)), y_pred_show, label=f'prediction', color='r')
    ax[0].plot(range(len(y_true_show)), y_true_show, label=f'true label', color='b')

    ax[1].plot(range(len(diff)),
               diff,
               label=f'diff {i}',
               color='y')
    ax[0].legend()
    ax[1].legend()
    # plt.pause(0.01)
    plt.show()
    fig.savefig(f'{res_file}.svg')
    plt.close()


def get_xgboost_feature(X_train, y_train):
    xgboost = XGBRegressor()
    xgboost.fit(X_train, y_train)
    features = xgboost.predict(X_train)
    return features
