import datetime

import numpy as np
import pandas as pd
import torch
from django.core.cache import cache

from .models import WaterInfo, AreaWeatherInfo
from lstm.utils import DependenceParser, create_custom_serializer, merge_dfs


class WaterInfoDependenceParser(DependenceParser):
    def __init__(self, json_data, datalength):
        super(WaterInfoDependenceParser, self).__init__(json_data)
        self.datalength = datalength
        self.predict_available = True
        self.times = datetime.datetime.now()

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

    data = scaler[0].transform(data)

    input_data = torch.tensor(data, dtype=torch.float32).to(device)

    output = model(input_data.unsqueeze(0))
    output = output.data.cpu().numpy()
    output = scaler[1].inverse_transform(np.array(output).reshape(-1, output.shape[-1]))
    return output
