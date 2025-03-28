import csv
import flask

app = flask.Flask(__name__)

csvfile = open('dataset.csv', 'r')
reader = csv.reader(csvfile)
next(reader)


@app.route('/api/now', methods=['GET'])
def get_now():
    return flask.jsonify(next(reader))


if __name__ == '__main__':
    app.run(debug=True)
    csvfile.close()
