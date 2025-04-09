from django.contrib import admin

from .models import WaterInfo, WaterPred


# Register your models here.
@admin.register(WaterInfo)
class WaterInfoAdmin(admin.ModelAdmin):
    list_display = (
        'times', 'temperature', 'humidity', 'winddirection', 'windpower',
        'rains', 'waterlevels63000120', 'rains63000100',
        'waterlevels63000100', 'waterlevels'
    )
    search_fields = ('times',)


@admin.register(WaterPred)
class WaterPredAdmin(admin.ModelAdmin):
    list_display = (
        "times", "waterlevel1", "waterlevel2", "waterlevel3",
        "waterlevel4", "waterlevel5", "waterlevel6"
    )
    search_fields = ('times',)
