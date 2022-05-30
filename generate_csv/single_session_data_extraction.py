from calendar import c
import json
import pandas as pd
import numpy as np
from generate_csv.raw_data_extract_func import *
from generate_features.common_columns_lists import *
from generate_features.features_functions import calc_delta_column


def read_json_objects(path):
    lst = []
    with open(path , 'r') as f:
        lst = f.readlines()
    return lst


def json2dict(path):
    try:
        json_obj_list = read_json_objects(path)
        data = json.loads(json_obj_list[0])
        return data
    except:
        return dict()

def single_session_hits_df(data):
    df = parse_hits_column(data)
    df['numOfProducts'] = df.apply(lambda x : len(x['hits.product']), axis=1)
    df['hits.time'] = df['hits.time'].astype(int)
    df = calc_delta_column(df, 'hits.time', 'hits.deltaTimeMS')
    return clear_df(df, hits_drop_columns)


def single_session_products_df(data):
    df = parse_products_column(data)
    return clear_df(df, products_drop_columns)


def single_session_tl_df(data):
    df0 = pd.DataFrame.from_dict(data, orient='index')
    df = df0.transpose()   
    for col in top_level_extract_columns:
        df_tmp = pd.json_normalize(df[col]).add_prefix(col+'.')
        df = pd.concat([df.drop(columns=[col]), df_tmp], axis=1)
    col = 'trafficSource.adwordsClickInfo'
    if col in df.columns.to_list():
        df0 = pd.json_normalize(df[col]).add_prefix(col+'.')
        df = pd.concat([df.drop(columns=[col]), df0], axis=1)
    return clear_df(df, top_level_drop_columns)



def generate_rawdata_df(path):
    data = json2dict(path)
    df_tl = single_session_tl_df(data)
    df_hits = single_session_hits_df(data)
    return df_tl, df_hits