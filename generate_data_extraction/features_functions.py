import pandas as pd
import numpy as np

def calc_delta_column(df, col, new_col):
    df[new_col] = df[col] - df[col].shift(fill_value=0)
    df[new_col] = df[new_col].apply(lambda x : 0 if x < 0 else x)
    return df


def columns_func(df, col, func):
    return func(df[col])



def column_z_score(df , col):
    s = np.std(df[col])
    m = np.mean(df[col])
    res = df[col].apply(lambda x: (x - m)/s if s > 0 else 0.0)
    return res


def columns_func_on_unique_values(df_gb, ind_col, col):
    df0 = df_gb[col].value_counts().reset_index(name='count').set_index(ind_col)
    df1 = df0[~df0.index.duplicated(keep='first')]
    records = list(df1.to_records(index=False))
    return pd.DataFrame(records, index=df1.columns, columns=[f'max.{col}'])


def filter_gte_columns_func(df, col, func, val):
    df[col] = df[df[col] >= val]
    return func(df[col])


def filter_in_columns_func(df, col, func, lst):
    df[col] = df[df[col] in lst]
    return func(df[col])


def filter_str_columns_func(df, col, func, reg):
    df[col] = df[df[col].str.contains(reg, case=False)]
    return func(df[col])


def convert_to_datetime(df, col):
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    return df


def convert_multiple_columns_to_numeric(df, cols):
    df[cols] = df[cols].apply(pd.to_numeric)
    return df


def group_by_start_or_end_of_month(df, datetime_col, data_col, agg_col, period_of_month):
    # period_of_month is in [SMS, SM]
    df0 = df.groupby([pd.Grouper(key=datetime_col, freq=period_of_month), data_col]).agg({agg_col : ['count']})
    df1 = df0.droplevel(1, axis=1).reset_index()
    return df1.groupby(data_col)[agg_col].agg('count')


def filter_by_weekends(df, datetime_col):
    df['weekDays'] = pd.to_datetime((df[datetime_col]).dt.date).dt.day_name()
    weekend = ['Saturday', 'Sunday']
    return df.loc[df['weekDays'].isin(weekend)]