from data_manager_tools import *
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl

# Read in the pandas dataframe
df = pd.read_csv(data_file)

# clean up the text in the description using part of the text_preprocess function
df = text_preprocess(df, stem=False, reduce=False, modify=False)

# feature engineer
df = feature_engineer(df, pos=False)

# for each unique job id aggregate the targets
df = aggregate_job_tags(df)

# convert data into numbers
x, y = df_to_values(df, preprocess=True)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.2, random_state=42)

# save data to file to easily run models on
pkl.dump((X_train, X_test, Y_train, Y_test), open(processed_data_file, 'wb'))
