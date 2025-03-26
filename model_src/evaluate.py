import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.backends.cuda import preferred_linalg_library
from torch.utils.data import DataLoader

import model.data_loader as data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, loss_fn, dataloader, metrics):
    model.eval()

    summ = []
    for data_batch, labels_batch in dataloader:
        data_batch = data_batch.to(device)
        labels_batch = labels_batch.to(device)

        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # show(output_batch, labels_batch)
        # print(output_batch.squeeze(-1), labels_batch)
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:010.7f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def show(output_batch, labels_batch):
    plt.plot(range(output_batch.shape[0]), [i[0] - j[0] + i[1] - j[1] for i, j in zip(output_batch, labels_batch)])
    plt.show()
