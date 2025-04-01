from django.contrib import admin
from django.contrib import messages
from django.db import transaction
from .models import LSTMModels


# Register your models here.
class LSTMModelAdmin(admin.ModelAdmin):
    list_display = ('date', 'name', 'rmse')

    # actions = ['delete_selected']

    def delete_selected(self, request, queryset):
        print('delete run')


admin.site.register(LSTMModels, LSTMModelAdmin)
