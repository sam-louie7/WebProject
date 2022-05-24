import numpy as np
from generate_data_extraction.features_functions import *
from generate_data_extraction.common_columns_lists import *


'''
REM : List of features

1. mean total time on session
2. mean products in session
3. mean time between hits 
4. ratio median / max time between hits
5. mean total page views
6. ratio of median / max products in hit
7. ratio viewed product / total products in session
8. mean num of view products                                   : hits.eCommerceAction.action_type = 2
9. ratio of median / max time between hits
10. mean time to add product
11. mean time to remove product
12. ratio of median / max add product
13. mean viewes producats
14. mean number of repeated web pages
15. max number of repeated web pages
16. mean time between web pages
17. ratio median / max time on max web pages
18. ** mean time spend on max viewd page



General Users Data measures :

0. buyers that are more then a minute + up tp a minute
1. time to perchuse histogram 
2. time to cart/ cart abandment hist
3. total time in session hist for buyers and non buyers
4. number of products in session hist for buyers and non buyers
5. time per product histogram
6. totals.transactions
7. totals.sessionQualityDim

'''





def hesitation_features(df_top_lvl, df_hits):

     # 1. mean total time on session should be high for hesitant buyers
    df0 = df_top_lvl.groupby('fullVisitorId').agg({'totals.timeOnSite': ['mean']}).droplevel(1, axis=1)
    df['avg_time_on_session'] = df0['totals.timeOnSite']
    
     # 2. mean products in session should be low for hesitant bayers
    df0 = df_group_by_mul_col(df_hits, 'numOfProducts', ['fullVisitorId', 'visitStartTime'], 'sum').reset_index(level=1)
    df1 = df0.groupby('fullVisitorId').agg({'numOfProducts': ['mean']}).droplevel(1, axis=1)
    df =  df1['numOfProducts'].to_frame('hits_num_products_per_session')
    
    # 3. mean time between hits should be high for hesitant bayers
    df0 = df_group_by_mul_col(df_hits, 'hits.deltaTimeMS', ['fullVisitorId', 'visitStartTime'], 'mean').reset_index(level=1)
    df1 = df0.groupby('fullVisitorId').agg({'hits.deltaTimeMS': ['mean']}).droplevel(1, axis=1)
    df['hits_avg_hits_time_diff'] = df1['hits.deltaTimeMS']
    
    # 4. ratio median / max time between hits
    df1 = df0.groupby('fullVisitorId').agg({'hits.deltaTimeMS': ['median', 'max']}).droplevel(1, axis=0)
    df['ratio_hits_time_diff'] = ratio_on_df_columns(df0['median'], df0['max'])
    
    # 5. ratio of unique / total page views should be low for hesitant buyers
    df0 = df_top_lvl.groupby('fullVisitorId').agg({'totals.pageviews': ['sum']}).droplevel(1, axis=1) # df0['totals.pageviews']
    df1 = 
    df['ratio_page_views'] = ratio_on_df_columns( , df0['totals.pageviews'])
    
    # MEAN FOR TOTALS VIEWS
    df0 = df_by_group_by_single_column(df_top_lvl, 'fullVisitorId', 'totals.hits', 'mean')
    df['avg_hits'] = df0['totals.hits']
    
    return df




def hesitation_hits_features(df_hits):
    
    
    
    # AVG_VIEW_PRD
    df0 = df_filter_and_group_by_mul_col(df_hits, 'hits.eCommerceAction.action_type', [2], ['fullVisitorId', 'visitStartTime'], 'count').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.eCommerceAction.action_type', 'mean')
    df['hits.avg_viewd_product'] = df1['hits.eCommerceAction.action_type']
    
    # AVG_TIME_ADD_PRD / RMV_PRD
    df0 = df_filter_and_agg_by_diff_col(df_hits, 'hits.eCommerceAction.action_type', [3], ['fullVisitorId', 'visitStartTime'], 'hits.time', 'mean').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.time', 'mean')
    df['hits.avg_time_add_prod'] = df1['hits.time']
    #
    df0 = df_filter_and_agg_by_diff_col(df_hits, 'hits.eCommerceAction.action_type', [4], ['fullVisitorId', 'visitStartTime'], 'hits.time', 'mean').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.time', 'mean')
    df['hits.avg_time_rmv_prod'] = df1['hits.time']

    return df
