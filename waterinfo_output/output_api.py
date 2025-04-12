import csv
from flask import Flask, request, Response, jsonify
from threading import Thread
import time
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
df = pd.read_csv("data.csv")

areaweather = df[["times", "temperature", "humidity", "winddirection", "windpower"]].copy()
areaweather["city"] = "杭州"
areaweather["county"] = "临安"
areaweather['times'] = pd.to_datetime(areaweather['times'])
areaweather = areaweather.sort_values(by='times', ascending=True)
areaweather['times'] = areaweather['times'].dt.strftime('%Y-%m-%d %H:%M:%S')

station63000110 = df[["times", "waterlevels63000120"]].copy()
station63000110.rename(columns={"waterlevels63000120": "waterlevels"}, inplace=True)
station63000110["rains"] = None
station63000110["station_id"] = "63000110"
station63000110['times'] = pd.to_datetime(station63000110['times'])
station63000110 = station63000110.sort_values(by='times', ascending=True)
station63000110['times'] = station63000110['times'].dt.strftime('%Y-%m-%d %H:%M:%S')

station63000100 = df[["times", "rains63000100", "waterlevels63000100"]].copy()
station63000100.rename(columns={"rains63000100": "rains", "waterlevels63000100": "waterlevels"}, inplace=True)
station63000100["station_id"] = "63000100"
station63000100['times'] = pd.to_datetime(station63000100['times'])
station63000100 = station63000100.sort_values(by='times', ascending=True)
station63000100['times'] = station63000100['times'].dt.strftime('%Y-%m-%d %H:%M:%S')

station63000200 = df[["times", "rains", "waterlevels"]].copy()
station63000200["station_id"] = "63000200"
station63000200['times'] = pd.to_datetime(station63000200['times'])
station63000200 = station63000200.sort_values(by='times', ascending=True)
station63000200['times'] = station63000200['times'].dt.strftime('%Y-%m-%d %H:%M:%S')

line = 0


def add():
    global line
    while True:
        time.sleep(5)
        if line < len(station63000100) - 1:
            line += 1


# 水站接口
@app.route('/api/waterinfo', methods=['GET'])
def get_waterinfo():
    station_id = request.args.get('station_id')
    if station_id == "63000100":
        return jsonify(station63000100.iloc[line].to_dict())
    elif station_id == "63000110":
        return jsonify(station63000110.iloc[line].to_dict())
    elif station_id == "63000200":
        return jsonify(station63000200.iloc[line].to_dict())
    else:
        return jsonify({'status': 'error', 'detail': 'station_id is invalid'}, status=400)


# 天气接口
@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    county = request.args.get('county')
    logging.info(f'city: {city}, county: {county}')
    if city == '杭州' and county == '临安':
        return jsonify(areaweather.iloc[line].to_dict())
    else:
        return jsonify({'status': 'error', 'detail': f'{city} or {county} is invalid'}, status_code=400)


if __name__ == '__main__':
    Thread(target=add, daemon=True).start()
    app.run(debug=True)
