import os
import gc
import numpy as np
import pandas as pd
# import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
from tempfile import NamedTemporaryFile


tf.debugging.set_log_device_placement(True)
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], False)

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.23
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

modelos = pd.read_csv('../models/models.csv', dtype=str)
sequencia_modelos = modelos['modelos'].tolist()


def prediz_sigmoide(matriz, sequencia_modelos):
    gpu = tf.config.list_physical_devices('GPU')
    tf.config.LogicalDeviceConfiguration(memory_limit=10000)
    logic = tf.config.list_logical_devices('GPU')

    vetor_sigmoide = []

    with tf.device('/device:GPU:0'):
        for modelo in sequencia_modelos:
            cnn = tf.keras.models.load_model('../models/'+ modelo)
            matriz_float = matriz.astype(float)
            sigma = cnn.predict(matriz_float / 255.0)

            sigma_squeeze = float(sigma)
            vetor_sigmoide.append(sigma_squeeze)

            tf.keras.backend.clear_session()

            gc.collect()

    return vetor_sigmoide

