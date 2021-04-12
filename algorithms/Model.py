import pickle
import numpy
import pandas
import sklearn
from pandas import DataFrame
from sklearn import preprocessing


class PredictionResult:
    def __init__(self, prediction, data):
        self.prediction = prediction
        self.data = data

    def to_dict(self):
        return {
            "prediction": self.prediction,
            "data": self.data
        }


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
        labels = []
        for column in data.columns:
            column_data = encoder.fit_transform(data[column])

            name_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))
            columns.append(column_data)
            labels.append(name_mapping)

        return DataFrame(self.columnsToRowsList(columns), columns=data.columns), labels

    def predict(self, predict_data, separator, predicting):
        predict_data = pandas.read_csv(predict_data, sep=separator)
        predict_data, labels = self.encodeLabels(predict_data)
        predict_data = predict_data.drop([predicting], 1)
        predictions = self.model.predict(predict_data)
        results = []
        predict_data_dict = predict_data.to_dict()
        for x in range(len(list(predict_data_dict))):
            current_labels = labels[x]
            dict_key = list(predict_data_dict)[x]
            for y in range(len(predict_data_dict[dict_key])):
                predict_data_dict[dict_key][y] = current_labels[predict_data_dict[dict_key][y]]
        for x in range(len(predictions)):
            data = {}
            key_list = list(predict_data_dict)
            for y in range(len(key_list)):
                data[key_list[y]] = str(predict_data_dict[key_list[y]][x])
            results.append(PredictionResult(predictions[x], data).to_dict())
        return results

    def train(self, file, separator, predicting_attribute):
        data = pandas.read_csv(file, sep=separator)

        data, labels = self.encodeLabels(data)

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
