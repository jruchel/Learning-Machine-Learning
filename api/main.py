from flask import Flask, json, request
from flask_cors import CORS, cross_origin
from sklearn.linear_model import LinearRegression
from topics.linear_regression import train
import traceback
import logging
import api


class LinearRegressionTrainingResult:
    def __init__(self, accuracy, intercept, coefficients):
        self.accuracy = accuracy
        self.intercept = intercept
        self.coefficients = coefficients


api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'


@api.route('/algorithms', methods=['GET'])
@cross_origin()
def algorithms():
    algos = ['linear-regression', 'some-algorithm']
    return json.dumps(algos)


@api.route('/algorithms/linear-regression', methods=['GET', 'POST'])
@cross_origin()
def linear_regression():
    try:
        arguments = request.args
        separator = arguments.get('separator')
        predicting = arguments.get('predicting')
        file = request.files['data']
        model, accuracy = train(file, separator, predicting, LinearRegression())
        intercept = model.intercept_
        coefficients = model.coef_
        return json.dumps(LinearRegressionTrainingResult(accuracy, intercept, coefficients.tolist()).__dict__)
    except Exception as error:
        return str(error)


if __name__ == "__main__":
    api.run(host='0.0.0.0')
    input("Press enter to exit")
