from flask import Flask, json, request
from flask_cors import CORS, cross_origin
import pandas as pd
from topics.linear_regression import encodeLabels, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy
from topics.linear_regression import train

api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'


@api.route('/algorithms/linear-regression', methods=['GET', 'POST'])
@cross_origin()
def linear_regression():
    arguments = request.args
    separator = arguments.get('separator')
    predicting = arguments.get('predicting')
    file = request.files['data']
    model, accuracy = train(file, separator, predicting, LinearRegression())

    return json.dumps(accuracy)


if __name__ == "__main__":
    api.run(host='0.0.0.0')
    input("Press enter to exit")
