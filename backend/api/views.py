import numpy as np
import torch
from django.http import Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import WaterInfo
from .serializers import PredictInputSerializer, PredictOutputSerializer, WaterInfoSerializer


class Water_Predict(APIView):
    def get(self, request):
        raise Http404

    def post(self, request):
        serializer = PredictInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = np.array(serializer.validated_data["features"])
        data = request.scaler[0].transform(data)

        input_data = torch.tensor(data, dtype=torch.float32).to(request.device)

        output = request.model(input_data.unsqueeze(0))
        output = output.data.cpu().numpy()
        output = request.scaler[1].inverse_transform(np.array(output).reshape(-1, output.shape[-1]))

        output_serializer = PredictOutputSerializer({'prediction': output.tolist()})
        return Response(output_serializer.data, status=status.HTTP_200_OK)


class Water_Info(APIView):
    def get(self, request):
        waterinfo = WaterInfo.objects.order_by("-times")[:18][::-1]
        waterinfo_serializer = WaterInfoSerializer(waterinfo, many=True)
        return Response(waterinfo_serializer.data, status=status.HTTP_200_OK)
