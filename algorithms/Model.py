import pickle
import numpy
import pandas
import sklearn
from pandas import DataFrame
from sklearn import preprocessing


class Model:

    def __init__(self, model):
        if isinstance(model, str):
            self.model = self.load(model)
        else:
            self.model = model

    def columnsToRowsList(self, columns):
        rows = []
        amount_of_rows = len(columns[0].tolist())

        for x in range(0, amount_of_rows):
            row = []
            for value in range(len(columns)):
                row.append(columns[value].tolist()[x])
            rows.append(row)
        return rows

    def encodeLabels(self, data):
        encoder = preprocessing.LabelEncoder()
        columns = []
        for column in data.columns:
            columns.append(encoder.fit_transform(data[column]))

        return DataFrame(self.columnsToRowsList(columns), columns=data.columns)

    def predict(self, predict_data, separator, predicting):
        predict_data = pandas.read_csv(predict_data, sep=separator)
        predict_data = self.encodeLabels(predict_data)
        predict_data = predict_data.drop([predicting], 1)
        predictions = self.model.predict(predict_data)
        results = []

        for x in range(len(predictions)):
            results.append((predictions[x], predict_data.iloc[x].to_dict()))
        return results

    def train(self, file, separator, predicting_attribute):
        data = pandas.read_csv(file, sep=separator)

        data = self.encodeLabels(data)

        x = numpy.array(data.drop([predicting_attribute], 1))
        y = numpy.array(data[predicting_attribute])

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

        self.model.fit(x_train, y_train)
        accuracy = self.model.score(x_test, y_test)
        self.model = self.model
        return accuracy

    def save(self, name):
        with open('{}.pickle'.format(name), "wb") as dump_file:
            pickle.dump(self.model, dump_file)

    def load(self, filename):
        pickle_in = open("{}.pickle".format(filename), 'rb')
        return pickle.load(pickle_in)

    def trimColumns(self, data, args):
        data_copy = data
        for arg in args:
            data_copy = data_copy.drop([arg], 1)
        return data_copy

    def get_model(self):
        return self.model
