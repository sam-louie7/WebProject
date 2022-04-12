import numpy as np
from generate_data_extraction.features_functions import *
from generate_data_extraction.ga_columns_lists import *


def top_level_ga_feature_vector(df):
    # columns list to convert to numeric
    df = convert_multiple_columns_to_numeric(df, cols=numeric_cols)
    groupby_df = df.groupby(['fullVisitorId'])
    # 1.
    df_cnt = groupby_df.apply(lambda x : columns_func(x, ['fullVisitorId'], np.count_nonzero))
    df_max = groupby_df.apply(lambda x : columns_func(x, totals_cols_1, np.max).add_prefix('max.'))
    df_avg = groupby_df.apply(lambda x : columns_func(x, totals_cols_1, np.average).add_prefix('avg.'))
    df_sum = groupby_df.apply(lambda x : columns_func(x, totals_cols_2, np.sum).add_prefix('sum.'))
    df_std = groupby_df.apply(lambda x : columns_func(x, totals_cols_2, np.std).add_prefix('std.'))
    # 2.
    df_dist = groupby_df.apply(lambda x : columns_dist(df, taffic_source_1, np.unique).add_prefix('dist.'))
    # 3.
    df_ts = groupby_df.apply(lambda x : filter2_and_columns_func(x, ['trafficSource.source'], len, taffic_source_values).add_prefix('tenen.'))
    regex_str = 'facebook|instagram|criteo|Newsletter'
    df_ss = groupby_df.apply(lambda x : filter3_and_columns_func(x, ['trafficSource.source'], len, regex_str).add_prefix('social.'))
    regex_str = 'yahoo|bing|baidu|google'
    df_se = groupby_df.apply(lambda x : filter3_and_columns_func(x, ['trafficSource.source'], len, regex_str).add_prefix('se.'))
    # 4.
    sr_tsr = df_ts['tenen.trafficSource.source'] / np.sum(df_ts['tenen.trafficSource.source']) # tenen_source_ratio
    sr_ssr = df_ss['social.trafficSource.source'] / np.sum(df_ss['social.trafficSource.source']) # social_source_ratio
    sr_ser = df_se['se.trafficSource.source'] / np.sum(df_se['se.trafficSource.source'])  # search_engine_source_ratio

    # 5
    regex_str = 'facebook|instagram|criteo|Newsletter'
    df_fb = groupby_df.apply(lambda x : filter3_and_columns_func(x, ['trafficSource.medium'], len, regex_str).add_prefix('fb.'))
    df_cpc = groupby_df.apply(lambda x : filter3_and_columns_func(x, ['trafficSource.medium'], len, 'cpc').add_prefix('cpc.'))
    regex_values = ['organic','referral','Email','Others']
    df_org = groupby_df.apply(lambda x : filter2_and_columns_func(x, ['trafficSource.medium'], len, regex_values).add_prefix('organic.'))

    # 6.
    sr_fbr = df_fb['fb.trafficSource.medium'] / np.sum(df_fb['fb.trafficSource.medium']) # facebook_medium_ratio
    sr_cpcr = df_cpc['cpc.trafficSource.medium'] / np.sum(df_cpc['cpc.trafficSource.medium']) #cpc_medium_ratio
    sr_orgr = df_org['organic.trafficSource.medium'] / np.sum(df_org['organic.trafficSource.medium']) # organic_medium_ratio

    # 7.



    df_conc = pd.concat([], axis=1)

