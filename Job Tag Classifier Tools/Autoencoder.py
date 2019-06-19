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

# make and load data
data_file = "D:/ML Data/Job Tag Classifier/job_data.csv"
X_train, X_test, _, _ = DataLoader(data_file, test_size=.25)

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
    conv = layers.Conv1D(10, 2, padding='same')(pool)
    pool = layers.MaxPooling1D(2, padding='same')(conv)

    flatten = layers.Flatten()(pool)

    dense = layers.Dense(300, activation="relu")(flatten)
    dense = layers.Dense(200, activation="relu")(dense)
    encoded = layers.Dense(100, activation="relu")(dense)
    dense = layers.Dense(200, activation="relu")(encoded)
    dense = layers.Dense(300, activation="relu")(dense)

    dense = layers.Dense(5760)(dense)
    reshape = layers.Reshape((int(5760/10), 10))(dense)


    conv = layers.Conv1D(10, 2, padding='same')(reshape)
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
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=5, batch_size=10, verbose=1)

# save the encoder half
model = encoder.save("Models/encoder_model")
