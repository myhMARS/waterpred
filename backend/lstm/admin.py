import joblib
import torch
from django.contrib import admin
from django.contrib import messages
from django.core.cache import cache
from django.db import transaction
from django.utils.safestring import mark_safe

from .models import LSTMModels, ScalerPT, TrainResult, PredictDependence


# Register your models here.
@admin.register(LSTMModels)
class LSTMModelAdmin(admin.ModelAdmin):
    list_display = ('date', 'name', 'md5', 'station_id', 'rmse', 'is_activate')
    list_filter = ('station__name',)
    ordering = ('rmse',)
    actions = ['delete_selected', 'activate_selected']

    def delete_selected(self, request, queryset):
        count = 0
        try:
            with transaction.atomic():
                for obj in queryset:
                    if cache.get(obj.station, None):
                        self.message_user(request, f'{obj.name} 模型正在运行，无法删除', messages.WARNING)
                    else:
                        obj.delete()
                        count += 1
            self.message_user(request, f'删除完成,共计删除{count}个模型', messages.SUCCESS)
        except Exception as e:
            self.message_user(request, f'删除失败{str(e)}', messages.ERROR)

    delete_selected.short_description = '删除所选模型（跳过正在运行的模型）'

    def activate_selected(self, request, queryset):
        activate_dict = dict()
        for obj in queryset:
            if not activate_dict.get(obj.station, None):
                activate_dict[obj.station] = obj
            else:
                self.message_user(request, f"站点{obj.station_id}同时不可启用多个模型", messages.ERROR)
                return

        for station, obj in activate_dict.items():
            if obj.is_activate:
                self.message_user(request, f'模型{obj.md5}-{obj.station_id}已启用', messages.INFO)
                continue
            runing_model_info = cache.get(obj.station_id)
            print(runing_model_info)
            LSTMModels.objects.filter(md5=runing_model_info['md5']).update(is_activate=False)
            self.message_user(request, f'模型{obj.md5}-{obj.station_id}已停用', messages.INFO)
            LSTMModels.objects.filter(md5=obj.md5).update(is_activate=True)
            cache.delete(obj.station)
            model = runing_model_info['model']
            device = runing_model_info['device']
            if device == torch.device("cuda"):
                model.load_state_dict(torch.load(obj.file))
            else:
                model.load_state_dict(torch.load(obj.file, map_location='cpu'))
            model.eval()
            scaler_model = ScalerPT.objects.get(lstm_model=obj.md5)
            scaler = joblib.load(scaler_model.file)

            runing_model_info['model'] = model
            runing_model_info['scaler'] = scaler
            runing_model_info['md5'] = obj.md5

            cache.set(obj.station_id, runing_model_info, timeout=None)

    activate_selected.short_description = '启用所选模型（只可选择一个）'


@admin.register(TrainResult)
class TrainResultAdmin(admin.ModelAdmin):
    list_display = ('lstm_model_id', 'display_image')
    list_filter = ('lstm_model__station__name',)
    readonly_fields = ("lstm_model_id", "image", "display_image")

    def display_image(self, obj):
        return mark_safe('<a href="{url}"><img src="{url}" width="240" height="80""/></a>'.format(
            url=obj.image.url
        ))

    search_fields = ("lstm_model__md5",)


@admin.register(PredictDependence)
class PredictDependenceAdmin(admin.ModelAdmin):
    list_display = ("station", "dependence")
    search_fields = ("station__id", "station__name")
