import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plot
import numpy
from matplotlib import style
import pandas
from colorama import Fore
import cmath
from sklearn.preprocessing import LabelEncoder

data = pandas.read_csv("StudentsPerformance.csv")


def encodeLabels(data):
    encoder = LabelEncoder()
    columns = []
    for column in data.columns:
        columns.append(encoder.fit_transform(data[column]))
    return pandas.DataFrame(
        list(zip(columns[0], columns[1], columns[2], columns[3], columns[4], columns[5], columns[6], columns[7])),
        columns=data.columns)


def trimColumns(data, args):
    data_copy = data
    for arg in args:
        data_copy = data_copy.drop([arg], 1)
    return data_copy


ignored_attributes = ["gender", "race/ethnicity", "lunch", "test preparation course", "math score", "reading score"]
predicting = 'writing score'

data = encodeLabels(data)
data = trimColumns(data, ignored_attributes)

x = numpy.array(data.drop([predicting], 1))
y = numpy.array(data[predicting])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("Accuracy: ", accuracy)

predictions = model.predict(x_test)

education_labels = ['Bachelor', 'Some College', 'Master', 'Associate', 'High School', 'Some High School']

x_axis = 'parental level of education'
style.use('ggplot')
plot.scatter(data[x_axis], data[predicting])
plot.xlabel(x_axis)
plot.ylabel(predicting)
plot.show()
