from rest_framework import permissions


class IsInAdminGroup(permissions.BasePermission):
    def has_permission(self, request, view):
        group_state = request.user.groups.filter(name='管理员组').exists() or request.user.is_superuser
        return request.user and request.user.is_authenticated and group_state
