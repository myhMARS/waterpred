from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated


class UserInfo(APIView):
    permission_classes = [IsAuthenticated,]

    def get(self, request):
        user = request.user

        return Response({
            'username': user.profile.name,
            'userid': user.username,
            'email': user.email,
            'phone': user.profile.phone,
            'manager': user.profile.manager,
            'location': user.profile.location,
        })
