from django.http import HttpResponse
from rest_framework.views import APIView

from .tasks import start_train


class TrainAPI(APIView):
    def get(self, request):
        # TODO: 测试api待处理
        start_train.delay()
        return HttpResponse("ok")
