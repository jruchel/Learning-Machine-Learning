import tensorflow as tf
import keras as kr
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pylab as plot
import pickle
from matplotlib import style

data = pd.read_csv('topics/students/student-mat.csv', sep=';')
data = data[
    ['G1',
     'G2',
     'G3',
     'studytime',
     'failures',
     'absences'
     ]
]

predict = 'G3'

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.1)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

print("Accuracy: \n", accuracy)
print("Coefficients: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

with open('studentmodel.pickle', "wb") as dump_file:
    pickle.dump(linear, dump_file)

pickle_in = open('studentmodel.pickle', 'rb')
linear = pickle.load(pickle_in)

p = 'studytime'
style.use('ggplot')
plot.scatter(data[p], data['G3'])
plot.xlabel(p)
plot.ylabel('Final Grade')
plot.show()
