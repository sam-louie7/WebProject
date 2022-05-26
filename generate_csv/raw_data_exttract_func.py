import dask.bag as db
import dask.dataframe as dd
import pandas as pd
from pandas import json_normalize


def low_level_df(f, obj):
    try:
        dbl = db.map(lambda x: f(x), obj).compute()
        df0 = pd.concat(dbl)
        return df0
    except:
        return pd.DataFrame()


def parse_hits_column(x):
    df = pd.DataFrame()
    if(len(x['hits']) > 0):
        df = pd.json_normalize(x, record_path='hits', record_prefix='hits.',
                               meta=['fullVisitorId', 'visitStartTime'])    
    return df


def parse_products_column(x):
    df = pd.DataFrame()
    if(len(x['hits']) > 0):
        df = pd.json_normalize(x, record_path=['hits', 'product'], record_prefix='hits.product.',
                               meta=['fullVisitorId', 'visitStartTime', ['hits', 'hitNumber']])
    return df



def general_dask_to_df__1(x, **kwargs):
    try:
        ddf = x.to_dataframe()
        df =  ddf.compute()
        df.reset_index(inplace=True)
        return df
    except:
        return pd.DataFrame()
    
def general_dask_to_df__2(x, **kwargs):
    try :
        extract_col = kwargs['col']
        ddf = (x.pluck(extract_col)).to_dataframe()
        df0 = ddf.compute()
        df0.reset_index(inplace=True)
        return df0.add_prefix(extract_col+'.')
    except:
        return pd.DataFrame()


def clear_df(df, columns_list):
    df = df.drop(columns=columns_list)
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df = df.replace(["", "/", ":", "(not set)"], "0").fillna("0").astype(str)
    df = df.dropna(how='all')
    return df




