import numpy as np
import torch
from django.core.cache import cache


def predict(data):
    scaler = cache.get('waterlevel_scaler')
    model = cache.get('waterlevel_model')
    model.lstm.flatten_parameters()

    device = cache.get('device')

    data = scaler[0].transform(data)

    input_data = torch.tensor(data, dtype=torch.float32).to(device)

    output = model(input_data.unsqueeze(0))
    output = output.data.cpu().numpy()
    output = scaler[1].inverse_transform(np.array(output).reshape(-1, output.shape[-1]))
    return output
