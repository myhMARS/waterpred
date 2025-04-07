import numpy as np
import pandas as pd
import torch
from django.core.cache import cache
from django.http import Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import WaterInfo
from .serializers import PredictInputSerializer, PredictOutputSerializer, WaterInfoSerializer


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
            waterinfo_serializer = WaterInfoSerializer(waterinfo, many=True)
            '''
            ["temperature", "humidity", "rains", "rains63000100",
            "windpower", "waterlevels63000100", "waterlevels63000120", "waterlevels"]
            '''
            data = pd.DataFrame(waterinfo_serializer.data)
            data = np.array(data[-12:])
            output = predict(data)
            return Response(waterinfo_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
