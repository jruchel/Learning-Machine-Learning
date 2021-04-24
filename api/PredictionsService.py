import base64
import pickle

from sklearn.neighbors import KNeighborsClassifier

from algorithms.Model import Model


def get_predictions_as_json_with_modelfile(modelfile, data, separator, predicting):
    modelfile = modelfile.replace("\\\\", "\\")
    modelfile = base64.b64decode(modelfile + '===')
    model_file = open("model.pickle", "wb")
    model_file.write(modelfile)
    model_file.close()
    model = Model(pickle.load(open("model.pickle", "rb")))
    return {"predictions": model.predict(data, separator, predicting)}


def get_knn_predictions_as_json(separator, predicting, neighbors, data):
    model = Model(KNeighborsClassifier(n_neighbors=neighbors))
    model.train(data, separator, predicting)
    data.seek(0)
    return {"predictions": model.predict(data, separator, predicting)}
