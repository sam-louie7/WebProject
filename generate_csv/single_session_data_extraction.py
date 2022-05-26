import json
import pandas as pd
from pandas import json_normalize
from raw_data_exttract_func import *
from generate_features.common_columns_lists import h


def json2dict(path):
    try:
        with open(path,'r') as f:
            data = json.loads(f.read())
        return data
    except:
        return None

def single_session_hits_df(data):
    df = parse_products_column(data)
    df['numOfProducts'] = df.apply(lambda x : len(x['hits.product']), axis=1)
    return clear_df(df, hits_drop_columns)


def single_session_products_df(data):
    df = parse_products_column(data)
    return clear_df(df, products_drop_columns)


def single_session_tl_df(data):
    df = pd.DataFrame.from_dict(data)
    for col in top_level_extract_columns:
        df0 = pd.json_normalize(df[col]).add_prefix(col+'.')
        