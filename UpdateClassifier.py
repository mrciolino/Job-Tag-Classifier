"""
Matthew Ciolino - Job Tag Classifier
Update the classifier when we have enough new data to process
"""

from keras import models
import sys

sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

# load pandas dataframe
data_file = "E:\ML Data\job_tag_classifier/big_bertha.csv"
X, _, Y, _ = DataLoader(data_file, test_size=0)

# encode the Input
encoder = models.load_model("Models/encoder_model.hd5")
X_encoded = encoder.predict(X)

# train the model
model = models.load_model("Models/classifier_model.hd5"")
model.fit(X_encoded, Y, epochs=5, batch_size=5)

# update weights
model.save("Models/model.hd5")
