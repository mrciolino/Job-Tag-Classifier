"""
Cutback.io
Our pipeline for tagging new jobs and optimizing the model as more data comes in
"""

from keras import models
import pandas as pd
import sys

sys.path.append("Job Tag Classifier Tools")
from Pipeline import *

# load models
model_file = "Models"
# model = models.load_model(model_file)

# read in new job to classifiy ----- Change to select new from job_data
sql_import_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data LIMIT 1 OFFSET 20;"]
X, _, Y, _ = DataLoader(sql_import_string, test_size=0)

# predict tags
#list_of_indices = model.predict(X)

# decode the target back into text - make sur excepts may rows of data
#predicition = tag_decoder(list_of_indices, threshold=.2)

# add the new data to a sql table to save for a training batch
sql_add_new_data_string = ["dbname='Cutback' host='127.0.0.1'", "new_data"]
update = batch_new_data(sql_import_string, sql_add_new_data_string)

# update the model and reset the new_data table
if update:
    update_classifier()
