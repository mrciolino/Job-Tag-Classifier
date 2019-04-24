"""
Matthew Ciolino - Job Tag Classifier
Our AI model that is used to predict job tags
"""
from keras import layers, backend
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np

from time import time
import sys
sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

# make sure keras sees our gpu
backend.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto(device_count={'GPU': 1})
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
backend.set_session(sess)

# ["dbname='Cutback' host='localhost' port='5432' user='postgres' password='1234'", "select * from job_data;"]
# ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
sql_string = ["dbname='Cutback' host='localhost' port='5432' user='postgres' password='1234'", "select * from job_data;"]
X_train, X_test, _, _ = DataLoader(sql_string, test_size=.1)

# tensorboard
tensorboard = TensorBoard(log_dir="Logs/autoencoder/{}".format(time()),
                          histogram_freq=1,
                          write_grads=True)

# reshape data to fit into our model
num_varibles = X_train.shape[1]
# 139,264
# look into large stride convs to reduce size

def model(num_varibles):

    input = layers.Input(shape=(num_varibles,))
    reshape = layers.Reshape((num_varibles, 1))(input)
    encoder_conv_1 = layers.Conv1D(250, 10, padding='same')(reshape)
    enconder_pool_1 = layers.MaxPooling1D(2, padding='same')(encoder_conv_1)
    encoder_conv_2 = layers.Conv1D(200, 8, padding='same')(enconder_pool_1)
    enconder_pool_2 = layers.MaxPooling1D(2, padding='same')(encoder_conv_2)
    encoder_conv_3 = layers.Conv1D(150, 6, padding='same')(enconder_pool_2)
    enconder_pool_3 = layers.MaxPooling1D(2, padding='same')(encoder_conv_3)
    encoder_conv_4 = layers.Conv1D(100, 4, padding='same')(enconder_pool_3)
    enconder_pool_4 = layers.MaxPooling1D(2, padding='same')(encoder_conv_4)
    encoder_conv_5 = layers.Conv1D(5, 2, padding='same')(enconder_pool_4)
    enconder_pool_5 = layers.MaxPooling1D(2, padding='same')(encoder_conv_5)

    flatten = layers.Flatten()(enconder_pool_5)

    dense = layers.Dense(500)(flatten)
    dense = layers.Dense(300)(dense)
    encoded = layers.Dense(100)(dense)
    dense = layers.Dense(300)(encoded)
    dense = layers.Dense(500)(dense)

    dense = layers.Dense(3200)(dense)
    reshape = layers.Reshape((640, 5))(dense)

    encoder_conv_1 = layers.Conv1D(5, 2, padding='same')(reshape)
    decoder_upsample_1 = layers.UpSampling1D(2)(encoder_conv_1)
    encoder_conv_2 = layers.Conv1D(100, 4, padding='same')(decoder_upsample_1)
    decoder_upsample_2 = layers.UpSampling1D(2)(encoder_conv_2)
    decoder_conv_3 = layers.Conv1D(150, 6, padding='same')(decoder_upsample_2)
    decoder_upsample_3 = layers.UpSampling1D(2)(decoder_conv_3)
    decoder_conv_4 = layers.Conv1D(200, 8, padding='same')(decoder_upsample_3)
    decoder_upsample_4 = layers.UpSampling1D(2)(decoder_conv_4)
    decoder_conv_5 = layers.Conv1D(250, 10, padding='same')(decoder_upsample_4)
    decoder_upsample_5 = layers.UpSampling1D(2)(decoder_conv_5)
    decoded = layers.Conv1D(1, 250, padding='same')(decoder_upsample_5)
    output = layers.Reshape((num_varibles,))(decoded)

    autoencoder = Model(inputs=input, outputs=output, name='Autoencoder')
    encoder = Model(inputs=input, outputs=encoded, name='Encoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder

# grab the network
autoencoder, encoder = model(num_varibles)
autoencoder.summary()

# fit the autoencoder
autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), epochs=1, batch_size=10, verbose=1, callbacks=[tensorboard])

# save the encoder half
model = encoder.save("Models/encoder_model")
