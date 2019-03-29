from keras.models import Model
from keras import layers
import numpy as np

import sys
sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

sql_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
X_train, X_test, _, _ = DataLoader(sql_string, test_size=.99)

# reshape data to fit into our model
num_varibles = X_test.shape[1]
X_test = X_test[:, :, np.newaxis]
X_test.shape


def model(num_varibles):

    input = layers.Input(shape=(num_varibles, 1))
    encoder_conv_1 = layers.Conv1D(250, 10, padding='same')(input)
    enconder_pool_1 = layers.MaxPooling1D(2, padding='same')(encoder_conv_1)
    encoder_conv_2 = layers.Conv1D(200, 8, padding='same')(enconder_pool_1)
    enconder_pool_2 = layers.MaxPooling1D(2, padding='same')(encoder_conv_2)
    encoder_conv_3 = layers.Conv1D(150, 6, padding='same')(enconder_pool_2)
    enconder_pool_3 = layers.MaxPooling1D(2, padding='same')(encoder_conv_3)

    dense = layers.Dense(500)(enconder_pool_3)
    dense = layers.Dense(400)(dense)
    dense = layers.Dense(300)(dense)

    encoded = layers.Dense(100)(dense)

    dense = layers.Dense(300)(encoded)
    dense = layers.Dense(400)(dense)
    dense = layers.Dense(500)(dense)

    decoder_conv_1 = layers.Conv1D(150, 6, padding='same')(dense)
    decoder_upsample_1 = layers.UpSampling1D(2)(decoder_conv_1)
    decoder_conv_2 = layers.Conv1D(200, 8, padding='same')(decoder_upsample_1)
    decoder_upsample_2 = layers.UpSampling1D(2)(decoder_conv_2)
    decoder_conv_3 = layers.Conv1D(250, 10, padding='same')(decoder_upsample_2)
    decoder_upsample_3 = layers.UpSampling1D(2)(decoder_conv_3)
    decoded = layers.Conv1D(1, 0, padding='same')(decoder_upsample_3)

    autoencoder = Model(inputs=input, outputs=decoded, name='AE')
    encoder = Model(inputs=input, outputs=encoded, name='encoder')

    autoencoder.compile(optimizer='adam', loss='mse')
    encoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


autoencoder, encoder = model(num_varibles)
autoencoder.summary()
encoder.summary()

autoencoder.fit(X_test, X_test, epochs=2, batch_size=10, verbose=1)
