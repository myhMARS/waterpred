import logging
import random

import torch
import numpy as np
import joblib
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, loss_fn, dataloader, metrics):
    model.train()
    summ = []
    with tqdm(total=len(dataloader)) as t:
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

            t.set_postfix(loss='{:010.7f}'.format(loss.item()))
            t.update()

    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:010.7f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, optimizer, loss_fn, train_dataloader, test_dataloader, metrics, epochs):
    best_val_rmse = 100

    for epoch in range(epochs):
        try:
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))

            # compute number of batches in one epoch (one full pass over the training set)
            train(model, optimizer, loss_fn, train_dataloader, metrics)
            # train(model, optimizer, loss_fn, test_dataloader, metrics)

            # Evaluate for one epoch on validation set
            val_metrics = evaluate(model, loss_fn, test_dataloader, metrics)

            val_rmse = val_metrics['RMSE']
            is_best = val_rmse <= best_val_rmse

            if is_best:
                logging.info("- Found new best RMSE at epoch {}/{}".format(epoch + 1, epochs))
                best_val_rmse = val_rmse
        except KeyboardInterrupt:
            break
    # evaluate_graph(model, test_dataloader)


def init_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    utils.set_logger()
    init_seed()

    logging.info("cuda is {}".format(torch.cuda.is_available()))

    scaler = (MinMaxScaler(), MinMaxScaler())
    seq_length = 12
    pred_length = 6
    train_dataset = data_loader.WaterLevelDataset(
        "dataset.csv",
        train=True,
        scaler=scaler,
        seq_length=seq_length,
        pred_length=pred_length,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    print(len(train_dataset))
    joblib.dump(scaler, "scaler.pkl")
    test_dataset = data_loader.WaterLevelDataset(
        "dataset.csv",
        train=False,
        scaler=scaler,
        seq_length=seq_length,
        pred_length=pred_length,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    input_size = 8
    hidden_size = 128
    output_size = pred_length
    model = net.Waterlevel_Model(input_size, hidden_size, output_size, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    metrics = net.metrics

    epochs = 200
    logging.info("Starting training for {} epoch(s)".format(epochs))
    train_and_evaluate(model, optimizer, loss_fn, train_dataloader, test_dataloader, metrics, epochs)
    torch.save(model.state_dict(), f'waterlevel_model_{input_size}_{hidden_size}_{output_size}.pt')
    model.load_state_dict(torch.load(f'waterlevel_model_{input_size}_{hidden_size}_{output_size}.pt'))
    model.eval()
    utils.show_action_data(model, train_dataloader, scaler=scaler[1], res_file=f'train_res_{seq_length}_{pred_length}')
    utils.show_action_data(model, test_dataloader, scaler=scaler[1], res_file=f'test_res_{seq_length}_{pred_length}')
