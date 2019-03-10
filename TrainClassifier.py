"""
Cutback.io
Our AI model that is used to predict job tags
"""

from sklearn.model_selection import GridSearchCV, train_test_split
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras import layers
from time import time
import sys

sys.path.append("Job Tag Classifier Tools")
from DataCollection import data_collection
from FeatureCreation import feature_creation
from FeatureProcessing import feature_processing


class DataLoader():

    def __init__(self, sql_string, num_words, test_size):
        super(DataLoader, self).__init__()
        print("Starting Data Collection")
        df = data_collection(sql_string)  # collect the data
        print("Starting Feature Creation")
        df = feature_creation(df)  # create some text features
        print("Starting Feature Processing")
        x, y = feature_processing(df, num_words)  # convert the text into numbers for processing
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42)  # validation
        return X_train, X_test, Y_train, Y_test


sql_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
num_words = 1e6
test_size = .2
X_train, X_test, Y_train, Y_test = DataLoader(sql_string, num_words, test_size)

# length of input/output
num_varibles = X_train.shape[1]
num_classes = Y_train.shape[1]

# open channel for TensorBoard
tensorboard = TensorBoard(log_dir="job_tag_classifier/logs/{}".format(time()),
                          histogram_freq=1,
                          write_grads=True)


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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy', 'precision', 'recall'])
    return model


# build the model
model = classification_model()

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=25, callbacks=[tensorboard])
# evaluate the model
model.evaluate(X_test, Y_test, verbose=0)
# save the best preformered model
model.save(model_file)
