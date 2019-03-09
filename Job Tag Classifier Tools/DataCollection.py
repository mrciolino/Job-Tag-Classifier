"""
Cutback.io
Collection of data collection functions that imports
and cleans our data before feature engineering
"""
import pandas as pd
import psycopg2
import sys

sql_connection_string = "dbname='Cutback' host='127.0.0.1'"


def load_data(sql_connection_string):
    try:
        conn = psycopg2.connect(sql_connection_string)
        sql = "select * from job_data;"
        data = pd.io.sql.read_sql_query(sql, conn)
        pd.io.sql.
        conn = None
    except:
        print("Unable to read sql database into pandas datafram")
        sys.exit(0)
    return df


def remove_empty_rows(df):
    try:
        df = df[df["job_description"].notnull()]
        df = df[df["job_title"].notnull()]
    except:
        print("Unable to remove empty row from the dataframe")
        sys.exit(0)


def data_collection(sql_connection_string):

    df = load_data(sql_connection_string)
    df = remove_empty_rows(df)

    return df
