import csv
import flask
from threading import Thread
import time

app = flask.Flask(__name__)


class WaterData:
    def __init__(self):
        self.file = open('data.csv', 'r')
        self.csvfile = csv.reader(self.file)
        next(self.csvfile)
        self.data = []

        self.run()

    def step(self):
        while True:
            self.data = next(self.csvfile)
            time.sleep(5)

    def run(self):
        thread = Thread(target=self.step)
        thread.daemon = True
        thread.start()


@app.route('/api/now', methods=['GET'])
def get_now():
    return flask.jsonify(datamanager.data)


if __name__ == '__main__':
    datamanager = WaterData()
    app.run(debug=False)
    datamanager.file.close()
