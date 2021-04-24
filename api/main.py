import base64
import json
import os
from flask import Flask, json, request
from flask_cors import CORS, cross_origin
from sklearn.neighbors import KNeighborsClassifier

from api.PredictionsService import get_predictions_as_json_with_modelfile, get_knn_predictions_as_json

from api.TrainingService import train_linear_regression

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


@api.route('/algorithms/predict', methods=['POST'])
@cross_origin()
def predict_with_model():
    try:
        modelfile = request.form['modelfile']
        data = request.files['data']
        separator = request.args['separator']
        predicting = request.args['predicting']
        response_data = get_predictions_as_json_with_modelfile(modelfile, data, separator, predicting)
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
        return create_response(get_knn_predictions_as_json(separator, predicting, neighbors, data), 200)
    except Exception as error:
        return create_response(str(error), 409)


@api.route('/algorithms/linear-regression', methods=['POST'])
@cross_origin()
def linear_regression():
    try:
        separator = request.args.get('separator')
        predicting = request.args.get('predicting')
        save = bool(request.args.get('save'))
        savename = request.args.get('savename')
        usersecret = request.args.get('usersecret')
        data = request.files['data']
        response_data = train_linear_regression(separator, predicting, save, savename, usersecret, data)
        return create_response(response_data, 200)
    except Exception as error:
        return create_response(str(error), 409)


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
