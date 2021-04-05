import sklearn
from sklearn.linear_model import LinearRegression
import numpy
import pandas
import topics.utils as utils

best = 0

model = utils.load('studentmodel.pickle')

print(utils.predict(model, [[0, 0, 18, 0, 0, 0, 4, 4, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 4, 1, 1, 3, 6, 2, 6]]))
