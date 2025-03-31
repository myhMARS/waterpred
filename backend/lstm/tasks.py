from celery import shared_task
from .train_src.train import train_lstm_model


@shared_task
def start_train():
    train_lstm_model()
