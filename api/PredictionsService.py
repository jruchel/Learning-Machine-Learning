import base64
import pickle

from algorithms.Model import Model


def get_predictions_as_json(modelfile, data, separator, predicting):
    modelfile = modelfile.replace("\\\\", "\\")
    modelfile = base64.b64decode(modelfile + '===')
    model_file = open("model.pickle", "wb")
    model_file.write(modelfile)
    model_file.close()
    model = Model(pickle.load(open("model.pickle", "rb")))
    return {"predictions": model.predict(data, separator, predicting)}
