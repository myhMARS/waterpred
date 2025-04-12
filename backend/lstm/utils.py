import pandas as pd
from rest_framework import serializers

from api.models import WaterInfo, AreaWeatherInfo


def merge_dfs(df_list):
    if not df_list:
        return pd.DataFrame()

    result = df_list[0]
    for i in range(1, len(df_list)):
        result = pd.merge(result, df_list[i], on="times", how="inner")
    return result


def create_custom_serializer(custom_model, custom_fields):
    class CustomSerializer(serializers.ModelSerializer):
        class Meta:
            model = custom_model
            fields = custom_fields

    return CustomSerializer


class DependenceParser:
    def __init__(self, json_data):
        dependence_data = json_data['dependence']

        self.weather_dependence = dependence_data['weather']
        self.stations_dependence = dependence_data['stations']
        self.target_station = json_data['station']
        self.target_field = dependence_data['target']

    def weather_dependence_parser(self):
        weather_custom_serializer = create_custom_serializer(
            AreaWeatherInfo,
            self.weather_dependence["fields"] + ["times"]
        )
        queryset = AreaWeatherInfo.objects.filter(
            county=self.weather_dependence["require"]
        ).order_by("times").all()
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
            ).order_by("times").all()
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
        ).order_by("times").all()
        target_res = target_custom_serializer(queryset, many=True)
        return pd.DataFrame(target_res.data)

    def time_series_split(self, data):
        data["times"] = pd.to_datetime(data["times"])
        data['time_diff'] = data["times"].diff()
        mask = data['time_diff'] >= pd.Timedelta(hours=2)
        split_indices = [0] + data.index[mask].tolist() + [len(data)]
        segment = [data.iloc[split_indices[i]: split_indices[i + 1]] for i in range(len(split_indices) - 1)]
        X, y = [], []
        for data in segment:
            if len(data) >= 12:
                X.append(data.drop(columns=["times", "time_diff"]))
                y.append(data[self.target_field])

        return X, y

    def get_dataset(self):
        weather_data = self.weather_dependence_parser()
        stations_data = self.stations_dependence_parser()
        target_data = self.target_dependence_parser()
        data = merge_dfs([weather_data, stations_data, target_data])

        X, y = self.time_series_split(data)
        train_availble = sum([len(i) for i in X]) > 3000

        input_size = len(X[0].columns)
        return train_availble, (X, y, input_size)
