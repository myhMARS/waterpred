import logging

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
        # station_id = request.query_params.get("station_id")
        # if not station_id:
        #     return Response({"detail": "prarms station_id required"}, status=status.HTTP_400_BAD_REQUEST)
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
        changinfo = ModelChangeSerializer(data=request.data)
        if not changinfo.is_valid():
            return Response(changinfo.errors, status=status.HTTP_400_BAD_REQUEST)
        try:
            cache_model = cache.get(changinfo.validated_data["station_id"], None)
            model_obj = LSTMModels.objects.get(
                md5=changinfo.validated_data["md5"],
                station_id=changinfo.validated_data["station_id"]
            )

            if cache_model:
                if cache_model["md5"] == changinfo.validated_data["md5"]:
                    return Response({"detail": "该模型正在运行中，无需重复启用"}, status=status.HTTP_200_OK)
                scaler = ScalerPT.objects.get(lstm_model=changinfo.validated_data["md5"])
                if cache_model['device'] == torch.device("cuda"):
                    cache_model["model"].load_state_dict(torch.load(model_obj.file))
                elif cache_model('device') == torch.device("cpu"):
                    cache_model["model"].load_state_dict(torch.load(model_obj.file, map_location='cpu'))
                cache_model['model'].eval()

                disabled_model_md5 = cache_model['md5']
                LSTMModels.objects.filter(md5=disabled_model_md5).update(is_activate=False)
                model_obj.is_activate = True
                model_obj.save()
                cache_model['md5'] = changinfo.validated_data["md5"]

                scaler = joblib.load(scaler.file)
                for _ in scaler:
                    _.feature_names_in_ = None
                cache_model['scaler'] = scaler

                cache.set(changinfo.validated_data["station_id"], cache_model, timeout=None)

                return Response({
                    'detail': 'success'
                },
                    status=status.HTTP_200_OK
                )
            else:
                return Response({
                    "detail": "站点未启用",
                },
                    status=status.HTTP_400_BAD_REQUEST
                )

        except LSTMModels.DoesNotExist:
            return Response({
                'detail': '未找到站点或模型'
            },
                status=status.HTTP_404_NOT_FOUND
            )

        except Exception as e:
            logging.info(e)
            return Response({
                'detail': f'服务器错误',
            },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelInfo(APIView):
    # permission_classes = [permissions.IsInAdminGroup]

    def get(self, request):
        station_id = request.query_params.get("station_id")
        if not station_id:
            return Response({"detail": "prarms station_id required"}, status=status.HTTP_400_BAD_REQUEST)
        model_info = cache.get(station_id, None)
        if model_info:
            return Response({
                'md5': model_info['md5'],
            },
                status=status.HTTP_200_OK
            )
        else:
            return Response({
                "detail": "未找到该站点模型"
            },
                status=status.HTTP_400_BAD_REQUEST
            )
