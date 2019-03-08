"""
Cutback.io
Collection of data collection functions that imports
and cleans our data before feature engineering
"""
import pandas as pd
import sys


def load_data(data_file):
    try:
        df = pd.read_csv(data_file)
    except:
        print("Unable to read data into pandas datafram")
        sys.exit(0)
    return df


def remove_empty_rows(df):
    try:
        df = df[df["job_description"].notnull()]
        df = df[df["job_title"].notnull()]
    except:
        print("Unable to remove empty row from the dataframe")
        sys.exit(0)
