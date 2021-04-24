import base64
import json
import os
import pickle

from flask import Flask, json, request
from flask_cors import CORS, cross_origin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from algorithms.Model import Model

api = Flask(__name__)
cors = CORS(api)
api.config['CORS_HEADERS'] = 'Content-Type'


@api.after_request
def cleanup(response):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        if f.endswith(".pickle"):
            os.remove(f)
    return response


@api.route('/algorithms/linear-regression/predict', methods=['POST'])
@api.route('/algorithms/predict', methods=['POST'])
@cross_origin()
def predict_with_model():
    try:
        model_file_bytes = request.form['modelfile']
        model_file_bytes = model_file_bytes.replace("\\\\", "\\")
        model_file_bytes = base64.b64decode(model_file_bytes + '===')
        data = request.files['data']
        separator = request.args['separator']
        predicting = request.args['predicting']
        model_file = open("model.pickle", "wb")
        model_file.write(model_file_bytes)
        model_file.close()
        model = Model(pickle.load(open("model.pickle", "rb")))
        response_data = {"predictions": model.predict(data, separator, predicting)}
        return create_response(response_data, 200)
    except Exception as error:
        return create_response(str(error), 409)


@api.route('/algorithms', methods=['GET'])
@cross_origin()
def algorithms():
    algos = ['linear-regression', 'k-nearest-neighbors']
    return create_response(algos, 200)


@api.route('/algorithms/k-nearest-neighbors', methods=['POST'])
@cross_origin()
def k_nearest_neighbours():
    try:
        arguments = request.args
        separator = arguments.get('separator')
        predicting = arguments.get('predicting')
        neighbors = int(arguments.get('neighbors'))
        data = request.files['data']
        model = Model(KNeighborsClassifier(n_neighbors=neighbors))
        model.train(data, separator, predicting)
        data.seek(0)
        response_data = {"predictions": model.predict(data, separator, predicting)}
        return create_response(response_data, 200)
    except Exception as error:
        return create_response(str(error), 409)


@api.route('/algorithms/linear-regression', methods=['POST'])
@cross_origin()
def linear_regression():
    try:
        arguments = request.args
        separator = arguments.get('separator')
        predicting = arguments.get('predicting')
        save = ''
        if arguments.get('save') == 'true':
            save = True
        else:
            save = False
        savename = ''
        if save is True:
            savename = "{}-{}".format(arguments.get('savename'), arguments.get('usersecret'))

        file = request.files['trainingData']
        regression = Model(LinearRegression())
        accuracy = regression.train(file, separator, predicting)
        intercept = regression.model.intercept_
        coefficients = regression.model.coef_
        if save:
            regression.save(savename)
            file_data = open("{}.pickle".format(savename), "rb").read()
            response_data = {
                "accuracy": accuracy,
                "intercept": intercept,
                "coefficients": coefficients.tolist(),
                "file": str(base64.b64encode(file_data)),
                "predicted": predicting
            }
        else:
            response_data = {
                "accuracy": accuracy,
                "intercept": intercept,
                "coefficients": coefficients.tolist(),
                "file": "",
                "predicted": predicting}
        return create_response(response_data, 200)
    except Exception as error:
        return create_response(str(error), 409)


@api.route('/read', methods=['POST'])
@cross_origin()
def read_from_database():
    return ""


@api.route('/save', methods=['POST'])
@cross_origin()
def save_to_database():
    return ""


def create_response(data, status_code):
    return api.response_class(response=json.dumps(data), status=status_code, mimetype='application/json')


def fit_neighbours(x_train, y_train, x_test, y_test):
    best_score = 0
    best_neighbours = 0
    for i in range(1, 100):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        if accuracy > best_score:
            best_score = accuracy
            best_neighbours = i
    return best_neighbours, best_score


if __name__ == "__main__":
    api.run(host='0.0.0.0')
    input("Press enter to exit")

# TODO make all endpoints return api.response
# TODO make make endpoints that return pieces of input data return full label names instead of numerical labels
