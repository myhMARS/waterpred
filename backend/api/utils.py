import base64
from datetime import datetime, timedelta
from io import BytesIO

import matplotlib

import numpy as np
import pandas as pd
import torch
from django.conf import settings
from django.core.cache import cache
from django.core.mail import EmailMessage
from django.template.loader import render_to_string

from lstm.utils import DependenceParser, create_custom_serializer, merge_dfs
from .models import WaterInfo, AreaWeatherInfo

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class WaterInfoDependenceParser(DependenceParser):
    def __init__(self, json_data, datalength):
        super(WaterInfoDependenceParser, self).__init__(json_data)
        self.datalength = datalength
        self.predict_available = True
        self.times = datetime.now()

    def weather_dependence_parser(self):
        weather_custom_serializer = create_custom_serializer(
            AreaWeatherInfo,
            self.weather_dependence["fields"] + ["times"]
        )
        queryset = AreaWeatherInfo.objects.filter(
            county=self.weather_dependence["require"]
        ).order_by("-times")[:self.datalength][::-1]
        weather_res = weather_custom_serializer(queryset, many=True)
        return pd.DataFrame(weather_res.data)

    def stations_dependence_parser(self):
        df_list = []
        for station_dependence in self.stations_dependence:
            water_custom_serializer = create_custom_serializer(
                WaterInfo,
                station_dependence["fields"] + ["times"]
            )

            queryset = WaterInfo.objects.filter(
                station=station_dependence["require"]
            ).order_by("-times")[:self.datalength][::1]
            water_res = water_custom_serializer(queryset, many=True)
            df = pd.DataFrame(water_res.data)
            times_col = df[['times']]
            other_cols = df.drop(columns=['times']).add_prefix(station_dependence["require"])
            result = pd.concat([times_col, other_cols], axis=1)
            df_list.append(result)
        merged_df = merge_dfs(df_list)
        return merged_df

    def target_dependence_parser(self):
        target_custom_serializer = create_custom_serializer(
            WaterInfo,
            [self.target_field, "times"]
        )
        queryset = WaterInfo.objects.filter(
            station=self.target_station
        ).order_by("-times")[:18][::-1]
        target_res = target_custom_serializer(queryset, many=True)
        self.times = target_res.data[-1]['times']
        return pd.DataFrame(target_res.data)

    def get_dataset(self):
        weather_data = self.weather_dependence_parser()
        stations_data = self.stations_dependence_parser()
        target_data = self.target_dependence_parser()
        data = merge_dfs([weather_data, stations_data, target_data]).head(12)

        X, y = self.time_series_split(data)
        predict_available = True if len(X) == 1 else False
        return predict_available, X[0].values if predict_available else None


def predict(station, data):
    model_obj = cache.get(station)
    device = model_obj['device']
    model = model_obj['model']
    model.eval()
    model.lstm.flatten_parameters()
    scaler = model_obj['scaler']

    data = np.array(data).reshape(-1, 1)
    data = scaler.transform(data)

    input_data = torch.tensor(data, dtype=torch.float32).to(device)

    output = model(input_data.unsqueeze(0))
    output = output.data.cpu().numpy()
    output = scaler.inverse_transform(np.array(output).reshape(-1, output.shape[-1]))
    return output


def sendWarning(times, station_name, waterlevels, warning_type, warning_waterlevel):
    html_context = get_html_context(times, station_name, waterlevels, warning_type, warning_waterlevel)
    msg = EmailMessage(
        subject='水情预警报告',
        body=html_context,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=['1972751073@qq.com', settings.DEFAULT_FROM_EMAIL]
    )
    msg.content_subtype = 'html'
    msg.send()


def generate_waterlevel_plot(timestamp, levels, warning_level):
    plt.rcParams['font.family'] = 'SimHei'  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    timestamps = [datetime.strftime(timestamp + timedelta(hours=i), "%Y-%m-%d %H:%M:%S") for i in range(1, 7)]
    plt.figure(figsize=(15, 4))
    plt.plot(timestamps, levels, marker='o', color='#3b82f6', label="预测水位")
    plt.axhline(y=warning_level, color='red', linestyle='--', label='预警水位')
    plt.title("未来6小时水位变化")
    plt.xlabel("时间")
    plt.ylabel("水位 (m)")
    plt.grid(True)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight')
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return img_base64


def get_html_context(timestamp, station_name, waterlevels, waring_type, warning_level):
    if len(waterlevels) == 1:
        html_context = render_to_string('warning_email.html', {
            "station_name": station_name,
            "timestamp": timestamp,
            "current_level": waterlevels[0],
            "warning_type": waring_type,
            "warning_level": warning_level

        })
    else:
        html_context = render_to_string('lstm_warning.html', {
            "station_name": station_name,
            "timestamp": timestamp,
            "predicted_level": f'{max(waterlevels):.2f}',
            "warning_type": waring_type,
            "warning_level": warning_level,
            "waterlevel_chart": generate_waterlevel_plot(timestamp, waterlevels, warning_level)
        })
    return html_context
