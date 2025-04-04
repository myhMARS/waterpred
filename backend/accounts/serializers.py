from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer
from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()


class CustomUserCreateSerializer(BaseUserCreateSerializer):
    class Meta(BaseUserCreateSerializer.Meta):
        model = User
        fields = ('username', 'email', 'password')  # 不包含 id 字段

    def create(self, validated_data):
        user = super().create(validated_data)
        return user

    def to_representation(self, instance):
        # 返回自定义字段（不含 id）
        return {
            "message": "注册成功",
            "user": {
                "username": instance.username,
            }
        }
