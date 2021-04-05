from sklearn import preprocessing
from pandas import DataFrame
import pickle
import pandas
import numpy
import sklearn


def predict(model, predict_data):
    predictions = model.predict(predict_data)
    results = []

    for x in range(len(predictions)):
        results.append((predictions[x], predict_data[x]))
    return results


def train(file, separator, predicting_attribute, model):
    data = pandas.read_csv(file, sep=separator)

    data = encodeLabels(data)

    x = numpy.array(data.drop([predicting_attribute], 1))
    y = numpy.array(data[predicting_attribute])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.9)

    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)

    return model, accuracy


def save(model):
    with open('studentmodel.pickle', "wb") as dump_file:
        pickle.dump(model, dump_file)


def load(model_file):
    pickle_in = open(model_file, 'rb')
    return pickle.load(pickle_in)


def encodeLabels(data):
    encoder = preprocessing.LabelEncoder()
    columns = []
    for column in data.columns:
        columns.append(encoder.fit_transform(data[column]))

    return DataFrame(columnsToRowsList(columns), columns=data.columns)


def columnsToRowsList(columns):
    rows = []
    amount_of_rows = len(columns[0].tolist())

    for x in range(0, amount_of_rows):
        row = []
        for value in range(len(columns)):
            row.append(columns[value].tolist()[x])
        rows.append(row)
    return rows


def trimColumns(data, args):
    data_copy = data
    for arg in args:
        data_copy = data_copy.drop([arg], 1)
    return data_copy
