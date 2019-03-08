from sklearn.model_selection import GridSearchCV
from keras.callbacks import TensorBoard
from keras.models import Sequential
from data_manager_tools import *
from keras import backend as K
from keras import layers
from time import time
import pickle as pkl

# making sure oru gpu is kicking
K.tensorflow_backend._get_available_gpus()

# import data
X_train, X_test, Y_train, Y_test = pkl.load(open(processed_data_file, 'rb'))

# length of input/output
num_varibles = X_train.shape[1]
num_classes = Y_train.shape[1]

# open channel for TensorBoard
tensorboard = TensorBoard(log_dir="job_tag_classifier/logs/{}".format(time()),
                          histogram_freq=1,
                          write_grads=True)

# grid search for hyper perameters
hyperparameters = {'learning_rate': [.001, .01, .1],
                   'dropout_rate': [.4, .6, .8],
                   'activation': ['relu', 'tanh', 'linear'],
                   'epochs': [25, 75, 150],
                   'batch_size': [1, 5, 20, 50],
                   'optimizer': ['Adam', 'SGD', 'RMSprop'], }

parameters = {'conv_layers': [1, 2, 3],
              'dense_layers': [1, 2, 3],
              'conv_nodes': [(100,10), (200,15), (300,20)],
              'dense_nodes': [50, 100, 150],
              'embedding_nodes': [25, 50, 75], }


def classification_model():
    # create model
    model = Sequential()
    model.add(layers.Embedding(input_dim=num_varibles + 1, output_dim=50, input_length=num_varibles))
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
model.summary()

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=25, callbacks=[tensorboard])

# evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

# save the best preformered model
model.save(model_file)
