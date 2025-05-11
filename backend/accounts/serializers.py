from rest_framework import serializers
from django.contrib.auth.models import User, Group
from djoser.serializers import UserCreateSerializer
from rest_framework.validators import UniqueValidator

from .models import Profile


class CustomEmailUniqueValidator(UniqueValidator):
    def __call__(self, value, serializer_field):
        self.message = f'邮箱 {value} 已存在'
        return super().__call__(value, serializer_field)


class CustomPhoneUniqueValidator(UniqueValidator):
    def __call__(self, value, serializer_field):
        self.message = f'手机号 {value} 已存在'
        return super().__call__(value, serializer_field)


class CustomUserCreateSerializer(UserCreateSerializer):
    email = serializers.EmailField(
        validators=[CustomEmailUniqueValidator(queryset=User.objects.all())]
    )
    phone = serializers.CharField(
        validators=[CustomPhoneUniqueValidator(queryset=Profile.objects.all())],
        required=True,
        write_only=True
    )
    name = serializers.CharField(
        required=True,
        write_only=True
    )

    class Meta(UserCreateSerializer.Meta):
        model = User
        fields = ('username', 'email', 'password', 'phone', 'name')

    def validate(self, attrs):
        phone = attrs.pop('phone')
        name = attrs.pop('name')
        super().validate(attrs)
        attrs['phone'] = phone
        attrs['name'] = name
        return attrs

    def create(self, validated_data):
        validated_data.pop('groups', None)
        phone = validated_data.pop('phone', None)
        name = validated_data.pop('name', None)
        user = super().create(validated_data)
        default_group, created = Group.objects.get_or_create(name='普通用户组')
        user.groups.add(default_group)
        profile = Profile(user=user, email=validated_data['email'], phone=phone, name=name)
        profile.save()
        return user

    def to_representation(self, instance):
        return {
            "message": "注册成功",
            "user": {
                "username": instance.username,
            }
        }
