import joblib
import torch
from django.contrib import admin
from django.contrib import messages
from django.core.cache import cache
from django.db import transaction

from .models import LSTMModels, ScalerPT


# Register your models here.
class LSTMModelAdmin(admin.ModelAdmin):
    list_display = ('date', 'name', 'rmse', 'is_running_model')
    ordering = ('rmse',)
    actions = ['delete_selected', 'activate_selected']

    def is_running_model(self, model):
        running_md5 = cache.get('model_md5')
        return 'running' if model.md5 == running_md5 else 'stop'

    is_running_model.short_description = '启用状态'

    def delete_selected(self, request, queryset):
        runing_model_md5 = cache.get('model_md5')
        count = 0
        try:
            with transaction.atomic():
                for obj in queryset:
                    if obj.md5 == runing_model_md5:
                        self.message_user(request, f'{obj.name} 模型正在运行，无法删除', messages.WARNING)
                    else:
                        obj.delete()
                        count += 1
            self.message_user(request, f'删除完成,共计删除{count}个模型', messages.SUCCESS)
        except Exception as e:
            self.message_user(request, f'删除失败{str(e)}', messages.ERROR)

    delete_selected.short_description = '删除所选模型（跳过正在运行的模型）'

    def activate_selected(self, request, queryset):
        count = queryset.count()
        if count == 1:
            obj = queryset.first()
            model_md5 = cache.get('model_md5')
            if obj.md5 != model_md5:
                model = cache.get('waterlevel_model')
                device = cache.get('device')
                if device == torch.device("cuda"):
                    model.load_state_dict(torch.load(obj.file))
                else:
                    model.load_state_dict(torch.load(obj.file, map_location='cpu'))
                model.eval()
                scaler_model = ScalerPT.objects.filter(lstm_model=obj.md5).first()
                scaler = joblib.load(scaler_model.file)

                cache.set('waterlevel_model', model, timeout=None)
                cache.set('waterlevel_scaler', scaler, timeout=None)
                cache.set('model_md5', obj.md5, timeout=None)
                self.message_user(request, '模型启用成功', messages.SUCCESS)
            else:
                self.message_user(request, '该模型已启用', messages.INFO)
        else:
            self.message_user(request, '只可启用一个模型', messages.ERROR)

    activate_selected.short_description = '启用所选模型（只可选择一个）'


admin.site.register(LSTMModels, LSTMModelAdmin)
