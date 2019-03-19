
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle
import sys
import os
sys.path.append("Job Tag Classifier Tools")
from Pipeline import DataLoader



# change data loader to use tokenizer so we get retrieve the column name

if not os.path.isfile("Data/data.pkl"):

    sql_string = ["dbname='Cutback' host='127.0.0.1'", "select * from job_data;"]
    X, _, Y, _ = DataLoader(sql_string, test_size=.2)

    with open("Data/data.pkl", 'wb') as vocab_file:
        pickle.dump((X, Y), vocab_file, protocol=pickle.HIGHEST_PROTOCOL)

else:

    with open("Data/data.pkl", 'rb') as handle:
        X, Y = pickle.load(handle)


select = SelectKBest(chi2, k=32)
X_new = select.fit_transform(X, Y)


mask = select.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

print(new_features)
