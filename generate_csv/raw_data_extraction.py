import multiprocessing as mp
import numpy as np
import pandas as pd
import json
import os
import sys
import datetime as dt

from dask.distributed import Client
from pandas import json_normalize
import dask.bag as db
import dask.dataframe as dd

from generate_features.common_columns_lists import *
from generate_csv.raw_data_extract_func import *

class BugRawJson:

    jsonlines = []
    col= 'trafficSource.adwordsClickInfo'
    time_col = 'visitStartTime'

    def read_json_objects(self, path):     
        with open(path , 'r') as f:
            self.jsonlines = f.readlines()


    def dask_bug_json_list(self):
        try: 
            daskobj = db.from_sequence(self.jsonlines, npartitions=mp.cpu_count()).map(json.loads)
            return daskobj
        except:
            return None
    


def top_level_df_creation(obj):
    col = 'trafficSource.adwordsClickInfo'
    df = general_dask_to_df__1(obj)
    df.reset_index(inplace=True)
    for c in top_level_extract_columns:
        df_tmp =  general_dask_to_df__2(obj, col=c)
        df = pd.concat([df.drop(columns=[c]), df_tmp], axis=1)
    df0 = pd.json_normalize(df[col]).add_prefix(col+'.')
    df = pd.concat([df.drop(columns=[col]), df0], axis=1)
    return clear_df(df, top_level_drop_columns)
    


def hits_df_creation(obj):
    df = low_level_df(parse_hits_column, obj)
    df['numOfProducts'] = df.apply(lambda x : len(x['hits.product']), axis=1)
    return clear_df(df, hits_drop_columns)


def products_df_creation(obj):
    df = low_level_df(parse_products_column, obj)
    return clear_df(df, products_drop_columns)

