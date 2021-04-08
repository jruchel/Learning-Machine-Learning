import sklearn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from termcolor import colored


def encodeLabels(data):
    labelEncoder = preprocessing.LabelEncoder()
    columns = data.columns
    columnData = []
    for column in columns:
        columnData.append(labelEncoder.fit_transform(list(data[column])))

    return pd.DataFrame(list(
        zip(columnData[0], columnData[1], columnData[2], columnData[3], columnData[4], columnData[5], columnData[6])),
        columns=columns)


def splitData(data, predict):
    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])
    return x, y


def createTestTrainGroups(x, y, test_size):
    return sklearn.model_selection.train_test_split(x, y, test_size=test_size)


def fit_neighbours(data, predict):
    x = np.array(data.drop([predict], 1))
    y = np.array(data[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
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


data = pd.read_csv('C:\\Users\\admin\Documents\\Learning-Machine-Learning\\data\\car.csv')

data = encodeLabels(data)

predict = 'class'

x, y = splitData(data, predict)

x_train, x_test, y_train, y_test = createTestTrainGroups(x, y, 0.1)

neighbours_count = fit_neighbours(x_train, y_train, x_test, y_test)[0]

model = KNeighborsClassifier(n_neighbors=neighbours_count)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)

predicted = model.predict(x_test)

for x in range(len(predicted)):
    prediction = predicted[x]
    data = x_test[x]
    actual = y_test[x]
    if prediction == actual:
        color = 'green'
    else:
        color = 'red'
    print(colored("Predicted: {}, Data: {}, Actual: {}".format(prediction, data, actual), color))
print("Accuracy: ", accuracy)
