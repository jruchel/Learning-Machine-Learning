from flask import Flask, json, request
from flask_cors import CORS, cross_origin
from sklearn.linear_model import LinearRegression
from algorithms.Model import Model
from sklearn.neighbors import KNeighborsClassifier
import json
import os
import pickle
import base64

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
@cross_origin()
def predict_linear_regression():
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
        return json.dumps(model.predict(data, separator, predicting))
    except Exception as error:
        return str(error)


@api.route('/algorithms', methods=['GET'])
@cross_origin()
def algorithms():
    algos = ['linear-regression', 'k-nearest-neighbours']
    return create_response(algos)


@api.route('/algorithms/k-nearest-neighbours', methods=['POST'])
@cross_origin()
def k_nearest_neighbours():
    try:
        arguments = request.args
        separator = arguments.get('separator')
        predicting = arguments.get('predicting')
        file = request.files['data']
        neighbours = arguments.get('neighbours')
        knn = Model(KNeighborsClassifier(n_neighbors=int(neighbours)))
        accuracy = knn.train(file, separator, predicting)
        return json.dumps({"accuracy": accuracy})
    except Exception as error:
        return str(error)


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
                "file": None,
                "predicted": predicting}
        return create_response(response_data)
    except Exception as error:
        return str(error)


@api.route('/read', methods=['POST'])
@cross_origin()
def read_from_database():
    return ""


@api.route('/save', methods=['POST'])
@cross_origin()
def save_to_database():
    return ""


def create_response(data):
    return api.response_class(response=json.dumps(data), status=200, mimetype='application/json')


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
