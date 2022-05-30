import pandas as pd
import numpy as np


def calc_delta_column(df, col, new_col):
    X = df[col] - df[col].shift(fill_value=0)
    df[new_col] = X.apply(lambda x : 0 if x < 0 else x)
    return df


def columns_func(df, col, func):
    return func(df[col])


def column_z_score(sr):
    s = np.std(sr)
    m = np.mean(sr)
    r = (sr - m) / s if m > 0 else pd.Series([0]*len(sr), index=sr.index.to_list())
    return r


def column_minmax_score(sr):
    n = np.min(sr)
    diff = np.max(sr) - np.min(sr)
    r = (sr - n) / diff
    return r


def columns_func_on_unique_values(df_gb, ind_col, col):
    df0 = df_gb[col].value_counts().reset_index(name='count').set_index(ind_col)
    df1 = df0[~df0.index.duplicated(keep='first')]
    records = list(df1.to_records(index=False))
    return pd.DataFrame(records, index=df1.columns, columns=[f'max.{col}'])



def filter_in_columns_func(df, col, func, lst):
    x = df[col].loc[df[col].isin(lst)]
    return func(x)


def filter_str_columns_func(df, col, func, reg):
    x = df[col].loc[df[col].str.contains(reg, case=False)]
    return func(x)


def convert_to_datetime(df, col):
    return pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')



def convert_multiple_columns_to_numeric(df, cols):
    df_cols = df.columns.to_list()
    for col in cols:
        if col in df_cols:
            df[col] = df[col].astype(int) 
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



'''
this method will be calculated in each column list of users
the vector will present the normal distribution of the values in each feature column 
'''


def convert_by_z(df, col_list) :
    def f(x, is_max):
        m = np.mean(x)
        s = np.std(x)
        h = (m + s) / s
        l =  (m - s) / s
        r = h if is_max else l
        return r
    v =  df.apply(lambda y: f(y, True) if y.name in col_list else f(y, False))
    return v 


def convert_by_minmax(df, col_list) :
    def f(x, is_max):
        m = np.max(x)
        n = np.min(x)
        r = m if is_max else n
        return r
    v =  df.apply(lambda y: f(y, True) if y.name in col_list else f(y, False))
    return v 


def corr_vector(df, vec):
    def f(n):
        if n >= 0.15:  # high corr to emotional = 1
            return 'emotional'
        elif n <= -0.15: # low corr to emotional , rational = 2
            return 'rational'
        else:  # no corr at all
            return 'nocorr'
    dft = df.T
    z = dft.apply(lambda x : x.corr(vec))
    res =  z.map(lambda x : f(x))
    return res



# DataFrame Utility functions
def df_filter_and_group_by_mul_col(df, col, f_values, grp_by_cols, func):
    df0 = df[df[col].isin(f_values)].groupby(grp_by_cols).agg({col : [func]}).droplevel(1, axis=1)
    return df0


def df_filter_and_group_by_single_col(df, col, f_value, grp_by_col, func):
    df0 = df[df[col] == f_value].groupby(grp_by_col).agg({col : [func]})
    return df0


def df_filter_and_agg_by_diff_col(df, filter_col, f_values, grp_by_cols, agg_col, func):
    df0 = df[df[filter_col].isin(f_values)].groupby(grp_by_cols).agg({agg_col : [func]}).droplevel(1, axis=1)
    return df0



def df_group_by_mul_col(df, col, grp_by_cols, func):
    df0 = df.groupby(grp_by_cols).agg({col : [func]}).droplevel(1, axis=1)
    return df0


def df_by_group_by_single_column(df, grp_by_col, agg_col, func):
    df0 = df.groupby(grp_by_col).agg({agg_col: [func]}).droplevel(1, axis=1)
    return df0


def ratio_on_df_columns(col1, col2):
    x = col1.divide(col2, fill_value=0.0)
    y = pd.DataFrame(x, index=col1.index, columns=['ratio'])
    r = y.loc[~np.isfinite(y['ratio']), 'ratio'] = 0.0
    return r
