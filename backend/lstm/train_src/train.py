import random
import logging  # TODO：日志输出到文件中

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from lstm.train_src.model_net import net, data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_and_evaluate(model, optimizer, loss_fn, train_dataloader, test_dataloader, metrics, epochs):
    best_val_rmse = 100

    for epoch in range(epochs):
        try:
            train(model, optimizer, loss_fn, train_dataloader, metrics)
            val_metrics = evaluate(model, loss_fn, test_dataloader, metrics)

            val_rmse = val_metrics['RMSE']
            is_best = val_rmse <= best_val_rmse

            if is_best:
                best_val_rmse = val_rmse
                logging.info("- Found new best RMSE at epoch {}/{}".format(epoch + 1, epochs))
        except Exception as e:
            logging.info("- Exception : {}".format(e))
            return 0
    return best_val_rmse


def train(model, optimizer, loss_fn, dataloader, metrics):
    model.train()
    summ = []
    for i, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)

        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            y_pred = y_pred.detach().cpu().numpy().squeeze()
            y_batch = y_batch.cpu().numpy()

            # y_pred = scaler[1].inverse_transform(np.array(y_pred).reshape(-1, 1))
            # y_batch = scaler[1].inverse_transform(np.array(y_batch).reshape(-1, 1))
            summary_batch = {metric: metrics[metric](y_pred, y_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:010.7f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


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



def train_lstm_model(data, input_size, hidden_size, output_size, time_stamp):
    seq_length = 12
    pred_length = output_size
    scaler = (MinMaxScaler(), MinMaxScaler())
    df = pd.DataFrame(data)
    try:
        train_dataset = data_loader.WaterLevelDataset(
            df=df,
            train=True,
            scaler=scaler,
            seq_length=seq_length,
            pred_length=pred_length,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)


        test_dataset = data_loader.WaterLevelDataset(
            df=df,
            train=False,
            scaler=scaler,
            seq_length=seq_length,
            pred_length=pred_length,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = net.Waterlevel_Model(input_size, hidden_size, output_size, 1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        loss_fn = nn.MSELoss()
        metrics = net.metrics
        epochs = 200
        rmse = train_and_evaluate(model, optimizer, loss_fn, train_dataloader, test_dataloader, metrics, epochs)
        if rmse:
            joblib.dump(scaler, f"temp/scaler_{time_stamp}.pkl")
            lstm_filename = f'waterlevel_model_{input_size}_{hidden_size}_{output_size}_{time_stamp}.pt'
            torch.save(model.state_dict(),
                       f'temp/{lstm_filename}')
    except Exception as e:
        logging.error("- Exception : {}".format(e))
        
