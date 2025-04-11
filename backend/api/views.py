import numpy as np
from django.http import Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import WaterInfo, WaterPred
from .serializers import PredictInputSerializer, PredictOutputSerializer, WaterInfoDataSerializer, \
    WaterInfoTimeSerializer, WaterPredDataSerializer
from .utils import predict


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

            waterpred = WaterPred.objects.order_by("-times").first()
            waterpred_data = WaterPredDataSerializer(waterpred)
            if None not in list(waterpred_data.data.values()):
                response_data["pred"] = list(waterpred_data.data.values())

            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
