"""
Cutback.io
Our pipeline for tagging new jobs and optimizing the model as more data comes in
"""
from data_manager_tools import *
from keras import models
import pandas as pd
import pickle

# load vocab
with open(feature_corpus_file, 'rb') as handle:
    tokenizer = pickle.load(handle)

# load models
model = models.load_model(model_file)

# read in new job to classifiy
df = pd.read_csv(new_data_file)

# handle new data -ignore words that are unseen
# add words to tokenzior when the next batch is run

# preprocess to convert text into matrix
new_job_features = process_new_data(df, tokenizer)

# predict tags
list_of_indices = model.predict(new_job_features)

# load target corpus
with open(target_corpus_file, 'rb') as handle:
    tokenizer = pickle.load(handle)


def tag_decoder(list_of_indices):
    threshold = .1
    target = []
    for i, num in enumerate(list_of_indices[0]):
        if num > threshold:
            target.append(str(tokenizer.classes_[i]))
    return target


print(tag_decoder(list_of_indices))
print(df.job_title[0])
print(df.job_tag_name[0])


# add new data to batch to be trained when it is full
# use our predicited tags as the target and hope accuracy is good enough
# fill batcha and re train models
# empty batch and restart process
