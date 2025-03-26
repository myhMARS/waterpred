import torch
import numpy as np
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from .serializers import InputSerializer, OutputSerializer


# Create your views here.

class Water_Info(APIView):
    def post(self, request):
        # 1. 验证输入数据
        serializer = InputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = np.array(serializer.validated_data["features"])
        data = request.scaler[0].transform(data)

        input_data = torch.tensor(data, dtype=torch.float32).to(request.device)

        # 3. 进行预测
        output = request.model(input_data.unsqueeze(0))
        output = output.data.cpu().numpy()
        output = request.scaler[1].inverse_transform(np.array(output).reshape(-1, output.shape[-1]))

        # 4. 序列化输出并返回 JSON 响应
        output_serializer = OutputSerializer({'prediction': output.tolist()})
        return Response(output_serializer.data, status=status.HTTP_200_OK)
