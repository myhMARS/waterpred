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
from accounts import permissions


class TrainAPI(APIView):
    def get(self, request):
        # TODO: 测试api待处理
        start_train.delay()
        return HttpResponse("ok")


class ModelList(APIView):
    # permission_classes = [permissions.IsInAdminGroup]

    def get(self, request):
        models = LSTMModels.objects.all()
        serializer = ModelListSerializer(models, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ChangeModel(APIView):
    # permission_classes = [permissions.IsInAdminGroup]

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
            cache_model.eval()

            disabled_model_md5 = cache.get('model_md5')
            LSTMModels.objects.filter(md5=disabled_model_md5).update(is_activate=False)
            LSTMModels.objects.filter(md5=md5serializer.validated_data["md5"]).update(is_activate=True)

            cache.set('waterlevel_model', cache_model, timeout=None)
            cache.set('model_md5', model.md5, timeout=None)
            scaler = joblib.load(scaler.file)
            for _ in scaler:
                _.feature_names_in_ = None
            cache.set('waterlevel_scaler', scaler, timeout=None)
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

        except Exception as e:
            return Response({
                'detail': f'服务器错误{e}',
            },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelInfo(APIView):
    # permission_classes = [permissions.IsInAdminGroup]

    def get(self, request):
        md5 = cache.get("model_md5")
        return Response({
            'md5': md5
        },
            status=status.HTTP_200_OK
        )
