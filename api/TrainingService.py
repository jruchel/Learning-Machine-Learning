import base64

from sklearn.linear_model import LinearRegression

from algorithms.Model import Model


def train_linear_regression(separator, predicting, save, savename, usersecret, data):
    if save is True:
        save = True
    else:
        save = False
    if save is True:
        savename = "{}-{}".format(savename, usersecret)

    regression = Model(LinearRegression())
    accuracy = regression.train(data, separator, predicting)
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
            "file": "",
            "predicted": predicting}
    return response_data
