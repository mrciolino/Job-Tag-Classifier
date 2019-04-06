"""
Matthew Ciolino - Job Tag Classifier
Update the classifier when we have enough new data to process
"""

from keras import models
import sys

sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

# load new_data sql table
# ["dbname='Cutback' host='localhost' port='5432' user='postgres' password='1234'", "select * from job_data;"]
# ["dbname='Cutback' host='127.0.0.1'", "select * from new_data;"]
sql_import_string = ["dbname='Cutback' host='127.0.0.1'", "select * from new_data"]
X, _, Y, _ = DataLoader(sql_import_string, test_size=0)

# encode the Input
encoder = models.load_model("Models/encoder_model.hd5")
X_encoded = encoder.predict(X)

# train the model
model = models.load_model("Models/classifier_model.hd5"")
model.fit(X_encoded, Y, epochs=5, batch_size=5)

# update weights
model.save("Models/model.hd5")
