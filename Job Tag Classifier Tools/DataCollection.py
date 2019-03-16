"""
Cutback.io
Collection of data collection functions that imports
and cleans our data before feature engineering
"""
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import traceback
import psycopg2
import sys


def load_data(sql_connection_string):
    try:
        connection = sql_connection_string[0]
        selection = sql_connection_string[1]
        conn = psycopg2.connect(connection)
        df = pd.io.sql.read_sql_query(selection, conn)
        conn.close()
    except:
        print("ERROR: Unable to read sql database into pandas dataframe")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)
    return df


def remove_empty_rows(df):
    try:
        df['job_description'].replace('', np.nan, inplace=True)
        df['job_title'].replace('', np.nan, inplace=True)
        df.dropna(subset=['job_description'], inplace=True)
        df.dropna(subset=['job_title'], inplace=True)
    except:
        print("ERROR: Unable to remove empty rows from the dataframe")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)
    return df


def add_new_data(df, sql_string):
    try:
        connection = sql_string[0]
        table = sql_string[1]
        engine = create_engine(connection)
        df.to_sql(name = table, con=engine, if_exists='append', index=False)
    except:
        print("ERROR: Unable to save new dataframe to sql table")
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)

    string = "select count(*) from " + table
    result = engine.execute(string)
    for row in result: length = row[0]
    engine.dispose()

    return True if length > 6 else False


def data_collection(sql_connection_string):

    df = load_data(sql_connection_string)
    df = remove_empty_rows(df)

    return df
