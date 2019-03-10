"""
Cutback.io
Collection of data collection functions that imports
and cleans our data before feature engineering
"""
import pandas as pd
import traceback
import psycopg2
import sys


def load_data(sql_connection_string):
    try:
        connection = sql_connection_string[0]
        selection = sql_connection_string[1]
        conn = psycopg2.connect(connection)
        df = pd.io.sql.read_sql_query(selection, conn)
        conn = None
    except:
        print("ERROR: Unable to read sql database into pandas dataframe")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)
    return df


def remove_empty_rows(df):
    try:
        df = df[df["job_description"].notnull()]
        df = df[df["job_title"].notnull()]
    except:
        print("ERROR: Unable to remove empty row from the dataframe")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)
    return df


def data_collection(sql_connection_string):

    df = load_data(sql_connection_string)
    df = remove_empty_rows(df)

    return df
