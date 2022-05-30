import numpy as np
from generate_features.features_functions import *


def init_top_lvl_df(df, numeric_cols):
    df['visitStartTime'] = convert_to_datetime(df, 'visitStartTime')
    df = convert_multiple_columns_to_numeric(df, cols=numeric_cols) # numeric_cols) : columns list to convert to int
    df = df.loc[df['totals.timeOnSite'] > 20] # take rows with timeOnSite > 10 sec
    return df



def init_hits_df(df, numeric_cols, time_col):
    df['visitStartTime'] = convert_to_datetime(df, 'visitStartTime')
    df = convert_multiple_columns_to_numeric(df, cols=numeric_cols)
    df = calc_delta_column(df, time_col, 'hits.deltaTimeMS')
    df = df.loc[df['totals.timeOnSite'] > 20] # take rows with timeOnSite > 10 sec
    return df