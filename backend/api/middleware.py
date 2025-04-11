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
        models_pt = LSTMModels.objects.filter(is_activate=True).all()
        for model_pt in models_pt:
            scaler_path = ScalerPT.objects.get(lstm_model=model_pt.md5)
            model = Waterlevel_Model(model_pt.input_size, model_pt.hidden_size, model_pt.output_size).to(device)
            if device == torch.device("cuda"):
                model.load_state_dict(torch.load(model_pt.file))
            else:
                model.load_state_dict(torch.load(model_pt.file, map_location='cpu'))
            model.eval()

            scaler = joblib.load(scaler_path.file)
            for _ in scaler:
                _.feature_names_in_ = None

            model_info_dict = {
                "model": model,
                "scaler": scaler,
                "md5": model_pt.md5,
                "device": device,
            }

            cache.set(model_pt.station_id, model_info_dict, timeout=None)

    def __call__(self, request):
        response = self.get_response(request)
        return response
