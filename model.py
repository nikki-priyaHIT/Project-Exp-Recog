from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
import logging
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialExpressionModel(object):
    logging.info("Model loaded successfully")
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        logging.info("Loading model from JSON file")
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
            logging.info("Model loaded successfully")
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        logging.info("Predicting emotion")
        global session
        set_session(session)
        logging.info("Session set")
        self.preds = self.loaded_model.predict(img)
        logging.info("Prediction done")
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
