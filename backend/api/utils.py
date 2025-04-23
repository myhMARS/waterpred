import base64
from datetime import datetime, timedelta
from io import BytesIO

import matplotlib
import numpy as np
import torch
from django.conf import settings
from django.core.cache import cache
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.contrib.auth.models import Group

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    admin_group = Group.objects.get(name='管理员组')
    admin_user = admin_group.user_set.exclude(email='').values_list('email', flat=True)
    msg = EmailMessage(
        subject='水情预警报告',
        body=html_context,
        from_email=settings.DEFAULT_FROM_EMAIL,
        to=list(admin_user)+settings.DEFAULT_FROM_EMAIL
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
