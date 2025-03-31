import torch
import torch.nn as nn
import joblib
from lstm.models import LSTMModels, ScalerPT
from lstm.train_src.model_net.net import Waterlevel_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

        # 1. 加载 PyTorch 模型（仅在 Django 启动时执行一次）
        self.model = Waterlevel_Model(8, 64, 6).to(device)
        if device == "cuda":
            self.model.load_state_dict(torch.load('./api/model/waterlevel_model_8_64_6.pt'))
        else:
            self.model.load_state_dict(torch.load('./api/model/waterlevel_model_8_64_6.pt', map_location='cpu'))
        self.model.eval()  # 设为评估模式，不会影响权重
        self.scaler = joblib.load('./api/model/scaler.pkl')
        for _ in self.scaler:
            _.feature_names_in_ = None

    def __call__(self, request):
        # 2. 将模型绑定到 request 对象，使其在整个请求过程中可用
        request.model = self.model
        request.scaler = self.scaler
        request.device = device

        # 处理请求
        response = self.get_response(request)
        return response
