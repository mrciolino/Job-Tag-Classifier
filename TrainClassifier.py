"""
Matthew Ciolino - Job Tag Classifier 
Our AI model that is used to predict job tags
"""
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import layers
from time import time
import sys

sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

sql_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
X_train, X_test, Y_train, Y_test = DataLoader(sql_string, test_size=.2)

# length of input/output
num_varibles = X_train.shape[1]
num_classes = Y_train.shape[1]

# open channel for TensorBoard
tensorboard = TensorBoard(log_dir="Logs/{}".format(time()),
                          histogram_freq=1,
                          write_grads=True)


def classification_model():
    # create model
    model = Sequential()
    model.add(layers.Embedding(input_dim=num_varibles + 1, output_dim=25, input_length=num_varibles))
    model.add(layers.Conv1D(200, 10, activation='relu'))
    model.add(layers.Conv1D(150, 10, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# build the model
model = classification_model()
print(model.summary())

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=25, callbacks=[tensorboard])

# evaluate the model
model.evaluate(X_test, Y_test, verbose=0)

# save the model
model.save("Models/model.hd5")
