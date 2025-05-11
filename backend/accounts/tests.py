from django.contrib.auth.models import User
from rest_framework.test import APITestCase
from rest_framework import status


class UserRegistrationTest(APITestCase):
    def test_user_registration_with_profile(self):
        url = '/auth/users/'
        user_data = {
            "username": "testuser",
            "password": "testpass123",
            "email": "testuser@example.com",
            "phone": "12345678901",
            "name": "测试用户"
        }

        response = self.client.post(url, user_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        user_exists = User.objects.filter(username='testuser').exists()
        self.assertTrue(user_exists)

        user = User.objects.get(username='testuser')
        self.assertIsNotNone(user.profile)
        self.assertEqual(user.profile.phone, "12345678901")
        self.assertEqual(user.profile.name, "测试用户")

        self.assertEqual(response.data, {
            "message": "注册成功",
            "user": {
                "username": "testuser"
            }
        })
