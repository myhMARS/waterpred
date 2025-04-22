import json

import joblib
import torch
from django.core.cache import cache
from django_celery_beat.models import PeriodicTask, IntervalSchedule

from lstm.models import LSTMModels, ScalerPT
from lstm.train_src.model_net.net import Waterlevel_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from backend.logging_config import get_task_logger

logger = get_task_logger(__name__)


class ModelMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.load_model()

    @staticmethod
    def load_model():
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
            scaler.feature_names_in_ = None

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


class WarningMessageTaskMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.init_task()

    def init_task(self):
        try:
            schedule, _ = IntervalSchedule.objects.get_or_create(
                every=3,
                period=IntervalSchedule.SECONDS,
            )

            PeriodicTask.objects.update_or_create(
                name='警告通知管理',
                defaults={
                    'interval': schedule,
                    'task': 'api.tasks.WarngingNoticeManage',
                    'args': json.dumps([]),
                }
            )
        except Exception as e:
            logger.error(e, self.__class__.__name__)

    def __call__(self, request):
        response = self.get_response(request)
        return response
