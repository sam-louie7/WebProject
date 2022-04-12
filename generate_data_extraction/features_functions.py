import pandas as pd
import numpy as np

def add_delta_column(df, col, new_col):
    df[new_col] = df[col] - df[col].shift(fill_value=0)
    return df


def columns_func(df, col, func):
    return func(df[col])


def columns_dist(df, col, func):
    return len(func(df[col]))


def filter_and_columns_func(df, col, func, val):
    df[col] = df[df[col] >= val]
    return func(df[col])


def filter2_and_columns_func(df, col, func, lst):
    df[col] = df[df[col] in lst]
    return func(df[col])


def filter3_and_columns_func(df, col, func, reg):
    df[col] = df[df[col].str.contains(reg, case=False)]
    return func(df[col])


def convert_to_datetime(df, col):
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    return df


def convert_multiple_columns_to_numeric(df, cols):
    df[cols] = df[cols].apply(pd.to_numeric)
    return df


def std_mean_to_max(std_col , max_col, avg_col):
    tmp_col = max_col.subtract(avg_col, fill_value=0)
    return tmp_col.devide(std_col).to_frame(name='std_mean_to_max').replace(np.inf, 1.0)

