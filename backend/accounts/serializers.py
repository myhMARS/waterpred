from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer

User = get_user_model()


class CustomUserCreateSerializer(BaseUserCreateSerializer):
    class Meta(BaseUserCreateSerializer.Meta):
        model = User
        fields = ('username', 'email', 'password')  # 不包含 id 字段

    def create(self, validated_data):
        validated_data.pop('groups', None)
        user = super().create(validated_data)
        default_group, created = Group.objects.get_or_create(name='普通用户组')
        user.groups.add(default_group)
        return user

    def to_representation(self, instance):
        # 返回自定义字段（不含 id）
        return {
            "message": "注册成功",
            "user": {
                "username": instance.username,
            }
        }
