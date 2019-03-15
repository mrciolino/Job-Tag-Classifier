"""
Cutback.io
Update the classifier when we have enough new data to process
"""
from keras import models
import sys

sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader

# load model weights
model_file = "Models"
model = models.load_model(model_file)

# load new_data sql table
sql_import_string = ["dbname='Cutback' host='127.0.0.1'", "select * from new_data"]
X, _, Y, _ = DataLoader(sql_import_string, test_size=0)

# train the model
model.fit(X_train, Y_train, epochs=10, batch_size=25)

# evaluate the model
model.evaluate(X_test, Y_test, verbose=0)

# update weights
model.save("Models/model.hd5")
