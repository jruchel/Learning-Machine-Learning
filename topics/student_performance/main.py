import sklearn
from sklearn.linear_model import LinearRegression
import numpy
import pandas
from sklearn.preprocessing import LabelEncoder


def encodeLabels(data):
    encoder = LabelEncoder()
    columns = []
    for column in data.columns:
        columns.append(encoder.fit_transform(data[column]))

    return pandas.DataFrame(columnListToRowList(columns), columns=data.columns)


def columnListToRowList(columns):
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


def main(file, separator, ignored_attributes, predicting_attribute, predict_data, test_size):
    data = pandas.read_csv(file, sep=separator)

    data = encodeLabels(data)

    data = trimColumns(data, ignored_attributes)

    x = numpy.array(data.drop([predicting_attribute], 1))
    y = numpy.array(data[predicting_attribute])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size)

    model = LinearRegression()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    predictions = model.predict(x_test)
    results = []
    for x in range(len(predictions)):
        results.append("Prediction: {}, Data: {}".format(predictions[x], x_test[x]))
    return results, accuracy


print(main("student-mat.csv", ';', [], 'G3', None, 0.1)[1])
