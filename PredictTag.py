"""
Cutback.io
Our pipeline for tagging new jobs and optimizing the model as more data comes in
"""
from keras import models
import pandas as pd
import pickle
import sys

sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader, tag_decoder

# load models
model_file =
model = models.load_model(model_file)

# read in new job to classifiy ----- Change to select new from job_data
sql_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
X, _, Y, _ = DataLoader(sql_string, test_size=0)

# predict tags
list_of_indices = model.predict(X)

# decode the target back into text
predicition = tag_decoder(list_of_indices, threshold = .2)





# add new data to batch to be trained when it is full
# use our predicited tags as the target and hope accuracy is good enough
# fill batcha and re train models
# empty batch and restart process
