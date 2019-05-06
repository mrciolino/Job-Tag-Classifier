"""
Matthew Ciolino - Job Tag Classifier
Our AI model that is used to predict job tags
"""
from keras import layers, backend
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
import tensorflow as tf
from time import time
import pickle as pkl
import numpy as np

import sys
sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

# make sure keras sees our gpu
backend.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto(device_count={'GPU': 1})
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
backend.set_session(sess)

# make and load data
data_file = "E:\ML Data\Cutback/big_bertha.csv"
X_train, X_test, _, _ = DataLoader(data_file, test_size=.1)

# tensorboard
tensorboard = TensorBoard(log_dir="Logs/autoencoder/{}".format(time()),
                          histogram_freq=1,
                          write_grads=True)

# reshape data to fit into our model
num_varibles = X_train.shape[1]

def model(num_varibles):

    input = layers.Input(shape=(num_varibles,))
    reshape = layers.Reshape((num_varibles, 1))(input)

    conv = layers.Conv1D(100, 10, padding='same')(reshape)
    pool = layers.MaxPooling1D(2, padding='same')(conv)
    conv = layers.Conv1D(80, 8, padding='same')(pool)
    pool = layers.MaxPooling1D(2, padding='same')(conv)
    conv = layers.Conv1D(60, 6, padding='same')(pool)
    pool = layers.MaxPooling1D(2, padding='same')(conv)
    conv = layers.Conv1D(40, 4, padding='same')(pool)
    pool = layers.MaxPooling1D(2, padding='same')(conv)
    conv = layers.Conv1D(20, 2, padding='same')(pool)
    pool = layers.MaxPooling1D(2, padding='same')(conv)
    conv = layers.Conv1D(3, 2, padding='same')(pool)
    pool = layers.MaxPooling1D(2, padding='same')(conv)

    flatten = layers.Flatten()(pool)

    dense = layers.Dense(300, activation="relu")(flatten)
    dense = layers.Dense(200, activation="relu")(dense)
    encoded = layers.Dense(100, activation="relu")(dense)
    dense = layers.Dense(200, activation="relu")(encoded)
    dense = layers.Dense(300, activation="relu")(dense)

    dense = layers.Dense(6528)(dense)
    reshape = layers.Reshape((int(6528/3), 3))(dense)

    conv = layers.Conv1D(3, 2, padding='same')(reshape)
    upsample = layers.UpSampling1D(2)(conv)
    conv = layers.Conv1D(20, 2, padding='same')(upsample)
    upsample = layers.UpSampling1D(2)(conv)
    conv = layers.Conv1D(40, 4, padding='same')(upsample)
    upsample = layers.UpSampling1D(2)(conv)
    conv = layers.Conv1D(60, 6, padding='same')(upsample)
    upsample = layers.UpSampling1D(2)(conv)
    conv = layers.Conv1D(80, 8, padding='same')(upsample)
    upsample = layers.UpSampling1D(2)(conv)
    conv = layers.Conv1D(100, 10, padding='same')(upsample)
    upsample = layers.UpSampling1D(2)(conv)
    conv = layers.Conv1D(1, 250, padding='same')(upsample)

    output = layers.Flatten()(conv)

    autoencoder = Model(inputs=input, outputs=output, name='Autoencoder')
    encoder = Model(inputs=input, outputs=encoded, name='Encoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder

# grab the network
autoencoder, encoder = model(num_varibles)
autoencoder.summary()

# fit the autoencoder
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=1, batch_size=100, verbose=1, callbacks=[tensorboard])

# save the encoder half
model = encoder.save("Models/encoder_model")
