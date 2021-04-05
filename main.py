from flask import Flask, json
from flask_cors import CORS, cross_origin
import pandas as pd
from topics.linear_regression import encodeLabels, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy

api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'


@api.route('/hello', methods=['GET'])
@cross_origin()
def sayHello():
    data = pd.read_csv('topics/students/student-mat.csv', sep=';')
    data = encodeLabels(data)
    predicting = 'G3'
    x = numpy.array(data.drop([predicting], 1))
    y = numpy.array(data[predicting])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return json.dumps("Linear regression accuracy: {}".format(model.score(x_test, y_test)))


if __name__ == "__main__":
    api.run(host='0.0.0.0')
    input("Press enter to exit")
