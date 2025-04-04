from django.core.cache import cache
from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
import torch

import joblib
from .models import LSTMModels, ScalerPT
from .serializers import ModelChangeSerializer, ModelListSerializer
from .tasks import start_train


class TrainAPI(APIView):
    def get(self, request):
        # TODO: 测试api待处理
        start_train.delay()
        return HttpResponse("ok")


class ModelList(APIView):
    def get(self, request):
        models = LSTMModels.objects.all()
        serializer = ModelListSerializer(models, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ChangeModel(APIView):
    def post(self, request):
        md5serializer = ModelChangeSerializer(data=request.data)
        if not md5serializer.is_valid():
            return Response(md5serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        try:
            cache_model = cache.get('waterlevel_model')
            model = LSTMModels.objects.get(md5=md5serializer.validated_data["md5"])
            scaler = ScalerPT.objects.get(lstm_model=md5serializer.validated_data["md5"])
            if cache.get('device') == torch.device("cuda"):
                cache_model.load_state_dict(torch.load(model.file))
            elif cache.get('device') == torch.device("cpu"):
                cache_model.load_state_dict(torch.load(model.file, map_location='cpu'))
            else:
                return Response({
                    'detail': 'device非法'
                },
                    status=status.HTTP_400_BAD_REQUEST
                )
            cache_model.to(cache.get('device'))
            cache_model.eval()
            cache.set('waterlevel_model', cache_model, timeout=None)
            cache.set('model_md5', model.md5, timeout=None)
            cache.set('waterlevel_scaler', joblib.load(scaler.file), timeout=None)
            return Response({
                "md5": cache.get('model_md5'),
            },
                status=status.HTTP_200_OK
            )

        except LSTMModels.DoesNotExist:
            return Response({
                'detail': '未找到指定MD5值的资源'
            },
                status=status.HTTP_404_NOT_FOUND
            )


class ModelInfo(APIView):
    def get(self, request):
        md5 = cache.get("model_md5")
        return Response({
            'md5': md5
        },
            status=status.HTTP_200_OK
        )
