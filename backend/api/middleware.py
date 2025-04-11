import joblib
import torch

from django.core.cache import cache

from lstm.models import LSTMModels, ScalerPT
from lstm.train_src.model_net.net import Waterlevel_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.load_model()

    def load_model(self):
        """从数据库加载模型和 scaler，并存入缓存"""
        model_pt = LSTMModels.objects.filter(is_activate=True).first()
        scaler_path = ScalerPT.objects.filter(lstm_model=model_pt.md5).first()

        model = Waterlevel_Model(8, 64, 6).to(device)
        if device == torch.device("cuda"):
            model.load_state_dict(torch.load(model_pt.file))
        else:
            model.load_state_dict(torch.load(model_pt.file, map_location='cpu'))
        model.eval()  # 设置为评估模式

        scaler = joblib.load(scaler_path.file)
        for _ in scaler:
            _.feature_names_in_ = None

        # 存入缓存（可以设定超时时间，单位秒）
        cache.set('waterlevel_model', model, timeout=None)
        cache.set('waterlevel_scaler', scaler, timeout=None)
        cache.set('model_md5', model_pt.md5, timeout=None)
        cache.set('device', device, timeout=None)

    def __call__(self, request):
        response = self.get_response(request)
        return response
