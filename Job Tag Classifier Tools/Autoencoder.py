from keras.callbacks import TensorBoard
from keras.models import Model
from keras import layers

import sys
sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

sql_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
X_train, X_test, _, _ = DataLoader(sql_string, test_size=.99)

# length of input/output
num_varibles = X_train.shape[1]

def model(num_varibles):

    input = layers.Input(shape=(num_varibles,None,None))
    encoder_conv_1 = layers.Conv1D(250, 10, padding='same')(input)
    enconder_pool_1 = layers.MaxPooling1D(2, padding='same')(encoder_conv_1)
    encoder_conv_2 = layers.Conv1D(200, 8, padding='same')(enconder_pool_1)
    enconder_pool_1 = layers.MaxPooling1D(2, padding='same')(encoder_conv_2)
    encoder_conv_3 = layers.Conv1D(150, 6, padding='same')(enconder_pool_1)
    encoded = layers.MaxPooling1D(2, padding='same')(encoder_conv_3)

    dense = layers.Dense(1500)(encoded)
    dense_encoded = layers.Dense(500)(dense)
    dense = layers.Dense(1500)(dense_encoded)

    decoder_conv_1 = layers.Conv1D(150, 6, padding='same')(dense)
    decoder_upsample_1 = layers.UpSampling1D(2)(decoder_conv_1)
    decoder_conv_2 = layers.Conv1D(200, 8, padding='same')(decoder_upsample_1)
    decoder_upsample_2 = layers.UpSampling1D(2)(decoder_conv_2)
    decoder_conv_3 = layers.Conv1D(250, 10, padding='same')(decoder_upsample_2)
    decoder_upsample_3 = layers.UpSampling1D(2)(decoder_conv_3)
    decoded = layers.Conv1D(1, 0, padding='same')(decoder_upsample_3)

    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

num_varibles
autoencoder = model(num_varibles)
autoencoder.summary()

autoencoder.fit(X_test, X_test, epochs=2, batch_size=10)
