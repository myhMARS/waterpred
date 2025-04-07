import numpy as np
import pandas as pd
import torch
from django.core.cache import cache
from django.http import Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import WaterInfo
from .serializers import PredictInputSerializer, PredictOutputSerializer, WaterInfoDataSerializer, WaterInfoTimeSerializer


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


class Water_Predict(APIView):
    def get(self, request):
        raise Http404

    def post(self, request):
        serializer = PredictInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = np.array(serializer.validated_data["features"])
        output = predict(data)

        output_serializer = PredictOutputSerializer({'prediction': output.tolist()})
        return Response(output_serializer.data, status=status.HTTP_200_OK)


class Water_Info(APIView):
    def get(self, request):
        try:
            waterinfo = WaterInfo.objects.order_by("-times")[:18][::-1]
            waterinfo_data = WaterInfoDataSerializer(waterinfo, many=True)

            waterinfo_time = WaterInfoTimeSerializer(waterinfo, many=True)
            times = []
            for _ in waterinfo_time.data:
                times.extend(_.values())
            response_data = {
                "times": times,
                "data": waterinfo_data.data,
            }
            if len(waterinfo_data.data) >= 12:
                data = pd.DataFrame(waterinfo_data.data)
                data = np.array(data[-12:])
                output = predict(data).tolist()[0]
                response_data['pred'] = output

            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
