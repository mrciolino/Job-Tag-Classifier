"""
Matthew Ciolino - Job Tag Classifier
Our AI model that is used to predict job tags
"""
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import layers, models
from time import time
import sys

sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

# ["dbname='Cutback' host='localhost' port='5432' user='postgres' password='1234'", "select * from job_data;"]
# ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
sql_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
X_train, X_test, Y_train, Y_test = DataLoader(sql_string, test_size=.2)

# open channel for TensorBoard
tensorboard = TensorBoard(log_dir="Logs/{}".format(time()),
                          histogram_freq=1,
                          write_grads=True)


def classification_model(num_varibles, num_classes):
    # create model
    model = Sequential()
    model.add(layers.Input(shape=(num_varibles,)))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(150, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# encode the Input
encoder = models.load_model("Models/encoder_model")
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# build the model
num_varibles = X_train_encoded.shape[1]
num_classes = Y_train.shape[1]
model = classification_model(num_varibles, num_classes)


# fit the model
model.fit(X_train_encoded, Y_train, validation_data=(X_test_encoded, Y_test), epochs=1, batch_size=1, callbacks=[tensorboard])

# evaluate the model
model.evaluate(X_test, Y_test, verbose=1)

# save the model
model.save("Models/classifier_model.hd5")
