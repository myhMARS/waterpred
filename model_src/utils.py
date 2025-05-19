import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tqdm import tqdm
import pandas as pd

# import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False
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
    res_y_pred, res_y_true = [], []
    for data_batch, labels_batch in dataloader:
        data_batch = data_batch.to(device)

        output_batch = model(data_batch)
        output_batch = output_batch.data.cpu().numpy()
        # print(output_batch)
        for output, labels in zip(output_batch, labels_batch):
            if scaler:
                output = scaler.inverse_transform(np.array(output).reshape(-1, output.shape[-1])).reshape(output.shape)
                labels = scaler.inverse_transform(np.array(labels).reshape(-1, output.shape[-1])).reshape(output.shape)
            # print(output)
            y_pred.append(np.array(output))
            # print(y_pred)
            y_true.append(np.array(labels))
            res_y_pred.extend(output)
            res_y_true.extend(labels)

    fig, ax = plt.subplots(4, 2)
    ymin = min(np.min(res_y_true), np.min(res_y_pred))
    ymax = max(np.max(res_y_true), np.max(res_y_pred))
    y_pred_show, y_true_show = [], []
    diff = []
    seq = [[[] for _ in range(2)] for _ in range(len(y_pred[0]))]
    with tqdm(total=len(y_pred)) as t:
        for i in range(len(y_pred)):
            y_true_show.append(y_true[i][0])
            y_pred_show.append(y_pred[i][0])
            diff.append(np.sum(np.abs(y_pred[i] - y_true[i])))
            # print((y_pred[i][0]-y_true[i][0])-(y_pred[i][1]-y_true[i][1]))

            for j in range(len(y_pred[i])):
                # print(y_true[i])
                seq[j][0].append(y_pred[i][j])
                seq[j][1].append(y_true[i][j])
                # break
            t.update()
    # print(seq)
    # for seq in seq:
    #     print(np.mean(seq))

    ax[0][0].set_ylim([ymin, ymax])

    ax[0][0].plot(range(len(y_pred_show)), y_pred_show, label=f'prediction', color='r')
    ax[0][0].plot(range(len(y_true_show)), y_true_show, label=f'true label', color='b')

    ax[0][1].plot(range(len(diff)),
                  diff,
                  label=f'diff {i}',
                  color='y')
    ax[0][0].legend()
    ax[0][1].legend()
    for i in range(1, len(y_pred[0]) // 2 + 1):
        for j in range(2):
            # print(seq[(i - 1) * 2 + j][])
            ax[i][j].plot(
                range(400),
                seq[(i - 1) * 2 + j][0][:400],
                label=f'seq_{(i - 1) * 2 + j}_pred',
                color='g'
            )
            ax[i][j].plot(
                range(400),
                seq[(i - 1) * 2 + j][1][:400],
                label=f'seq_{(i - 1) * 2 + j}_true',
                color='b'
            )
            ax[i][j].legend()
    plt.show()
    fig.savefig(f'{res_file}.svg')
    plt.close()


def roll_predict(model, sc, device):
    df = pd.read_csv('data.csv')
    dataset = df['waterlevels'].values

    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots(figsize=(10, 30))

    for i in range(len(dataset) - 18):  # 12输入 + 最多6步预测
        # 1. 获取模型输入
        X = dataset[i:i + 12].reshape(-1, 1)
        X_scaled = sc.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        y_pred_list = X_tensor.cpu().detach().numpy()  # 确保在 GPU 上的张量被移到 CPU

        for j in range(6):
            with torch.no_grad():
                input_data = torch.tensor(y_pred_list[:, j:, :], dtype=torch.float32).to(device)
                y_pred = model(input_data).cpu().numpy()
                y_pred = y_pred.reshape(1, 1, 1)
                y_pred_list = np.append(y_pred_list, y_pred, axis=1)
        y_pred_list = y_pred_list.squeeze().tolist()
        # y_pred_list = sc.inverse_transform(y_pred_list)
        y_true = sc.transform(dataset[i + 12:i + 12 + 6].reshape(-1,1)).squeeze()
        # print(y_pred_list[12:],y_true)
        # 4. 清空并画图
        ax.cla()
        print(max(y_pred_list)-min(y_pred_list))
        ax.plot(range(6), y_pred_list[12:], label='预测值', color='r')
        ax.plot(range(len(y_true)), y_true, label='真实值', color='b')
        ax.set_title(f"Step: {i}")
        ax.set_ylim(0, 1)
        ax.legend()

        plt.pause(0.1)

    plt.ioff()
    plt.show()
