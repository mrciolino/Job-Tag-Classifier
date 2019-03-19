"""
Matthew Ciolino - Job Tag Classifier 
A pipeline for operating the model from one function
"""

from sklearn.model_selection import train_test_split
import pickle
import sys

sys.path.append("Job Tag Classifier Tools")
from DataCollection import data_collection, add_new_data
from FeatureCreation import feature_creation
from FeatureProcessing import feature_processing


def DataLoader(sql_string, test_size):

    print("Starting Data Collection")
    df = data_collection(sql_string)  # collect the data

    print("Starting Feature Creation")
    df = feature_creation(df)  # create some text features

    print("Starting Feature Processing")
    x, y = feature_processing(df)  # convert the text into numbers for processing

    print("Data Loading Compelete")
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=42)  # validation

    return X_train, X_test, Y_train, Y_test


def BatchData(sql_import_string, sql_add_new_data_string):

    # import the new data we are prediciting
    df = data_collection(sql_import_string)

    # save that data back into the new data table
    update = add_new_data(df, sql_add_new_data_string)

    return update


def tag_decoder(list_of_indices, threshold):

    with open("Models/Tokenizers/target_tokens.pkl", 'rb') as handle:
        tokenizer = pickle.load(handle)

    target = []
    for i, num in enumerate(list_of_indices[0]):
        if num > threshold:
            target.append(str(tokenizer.classes_[i]))
    return target
