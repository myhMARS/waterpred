import torch
import numpy as np
import joblib
from model.net import Waterlevel_Model

model = Waterlevel_Model(8, 64, 6)
model.load_state_dict(torch.load('Waterlevel_model.pt'))
model.eval()

scaler = joblib.load("scaler.pkl")


def predict(data):
    data = np.array(data)
