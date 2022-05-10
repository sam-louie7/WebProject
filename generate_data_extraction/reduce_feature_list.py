import numpy as np
from generate_data_extraction.features_functions import *
from generate_data_extraction.common_columns_lists import *


'''
REM : List of features from Tal's work ,  i will start with this and see how it goes

pr_TENENG_SOURCE_RATIO
pr_SOCIAL_SOURCE_RATIO
pr_SE_SOURCE_RATIO
pr_SOURCE_MEDIUM_FB_RATIO
pr_SOURCE_MEDIUM_CPC_RATIO
pr_SOURCE_MEDIUM_ORGANIC_RATIO
#
pr_AVG_VIEW_PRD                             : hits.eCommerceAction.action_type = 2
pr_AVG_PRODUCTS_IN_SESSION  
pr_AVG_AVG_TIMETOHIT                        : hits.deltaTimeMS
pr_AVG_TIME_ADD_PRD                          
pr_AVG_PAYMENT_PAGE                         : hits.contentGroup.contentGroup1   
pr_AVG_DROPDOWN_CLICK                       : hits.contentGroup.contentGroup1    
pr_AVG_SUPPORT_PAGE                         : hits.contentGroup.contentGroup1    
pr_AVG_FEEDBACK_PAGE                        : hits.contentGroup.contentGroup1    
pr_AVG_SHOPPING_CART                        : hits.contentGroup.contentGroup1    
pr_AVG_DROP_FILTER_CLICK                    : hits.contentGroup.contentGroup1    
pr_AVG_CNT_SEARCHKEYWORD                    : hits.contentGroup.contentGroup1    
#
pr_weekend_ratio                            : Sat-Sun ratio   
pr_endmonth_ratio                           : end of month 15-30 in the month 
pr_HOUR_CAT1_ratio                          : night time 20:00 to 04:00

'''



def init_top_lvl_df(df, numeric_cols):
    df['visitStartTime'] = convert_to_datetime(df, 'visitStartTime')
    df = convert_multiple_columns_to_numeric(df, cols=numeric_cols) # numeric_cols) : columns list to convert to int
    df = df.loc[df['totals.timeOnSite'] > 10] # take rows with timeOnSite > 10 sec
    return df



def init_hits_df(df, numeric_cols, time_col):
    df['visitStartTime'] = convert_to_datetime(df, 'visitStartTime')
    df = convert_multiple_columns_to_numeric(df, cols=numeric_cols)
    df = calc_delta_column(df, time_col, 'hits.deltaTimeMS')
    df = df.loc[df['totals.timeOnSite'] > 10] # take rows with timeOnSite > 10 sec
    return df



def emotional_rational_hits_features(df_hits):
    # AVG_PRODUCTS_IN_SESSION
    df0 = df_group_by_mul_col(df_hits, 'numOfProducts', ['fullVisitorId', 'visitStartTime'], 'sum').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'numOfProducts' , 'mean')
    df =  column_z_score(df1['numOfProducts'].fillna(0)).to_frame('hits.products_per_session')
    # AVG_VIEW_PRD
    df0 = df_filter_and_group_by_mul_col(df_hits, 'hits.eCommerceAction.action_type', [2], ['fullVisitorId', 'visitStartTime'], 'count').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.eCommerceAction.action_type', 'mean') 
    df['hits.avg_viewd_product'] = column_z_score(df1['hits.eCommerceAction.action_type'].fillna(0))
    # AVG_AVG_TIMETOHIT
    df0 = df_group_by_mul_col(df_hits, 'hits.deltaTimeMS', ['fullVisitorId', 'visitStartTime'], 'mean').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.deltaTimeMS', 'mean')
    df['hits.avg_time2hit'] = column_z_score(df1['hits.deltaTimeMS'].fillna(0))
    # AVG_TIME_ADD_PRD / RMV_PRD
    df0 = df_filter_and_agg_by_diff_col(df_hits, 'hits.eCommerceAction.action_type', [3], ['fullVisitorId', 'visitStartTime'], 'hits.time', 'mean').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.time', 'mean')
    df['hits.avg_time_add_prod'] = column_z_score(df1['hits.time'].fillna(0))
    #
    df0 = df_filter_and_agg_by_diff_col(df_hits, 'hits.eCommerceAction.action_type', [4], ['fullVisitorId', 'visitStartTime'], 'hits.time', 'mean').reset_index(level=1)
    df1 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.time', 'mean')
    df['hits.avg_time_rmv_prod'] = column_z_score(df1['hits.time'].fillna(0))
    
    # 
    df0 = df_hits.groupby(['fullVisitorId', 'hits.contentGroup.contentGroup1'])['hits.contentGroup.contentGroup1'].agg('count').to_frame('contentGroupCount').reset_index(level=1)
    # SUPPORT_PAGE
    df1 = df0.loc[df0['hits.contentGroup.contentGroup1'] ==  'Support Page']
    df['hits.support_page'] = column_z_score(df1['contentGroupCount'].fillna(0))
    #  Article_PAGE
    df1 = df0.loc[df0['hits.contentGroup.contentGroup1'] ==  'Article Page']
    df['hits.article_page'] = column_z_score(df1['contentGroupCount'].fillna(0))
    # Thank you page
    df1 = df0.loc[df0['hits.contentGroup.contentGroup1'] ==  'Thank you Page']
    df['hits.search_keyword'] = column_z_score(df1['contentGroupCount'].fillna(0))
    # product page
    df1 = df0.loc[df0['hits.contentGroup.contentGroup1'] ==  'Product Page']
    df['hits.product_page'] = column_z_score(df1['contentGroupCount'].fillna(0))
     # Category Page
    df1 = df0.loc[df0['hits.contentGroup.contentGroup1'] ==  'Category Page']
    df['hits.category_page'] = column_z_score(df1['contentGroupCount'].fillna(0))

    return df


def emotional_rational_toplvl_features(df_top_lvl):
    # tenen source ratio :
    sr = df_top_lvl.groupby('fullVisitorId').apply(lambda x : filter_in_columns_func(x, 'trafficSource.source', len, traffic_source_values))
    df = column_z_score(sr.fillna(0)).to_frame('trafficSourceRatio')
    # social source ratio :
    sr= df_top_lvl.groupby('fullVisitorId').apply(lambda x : filter_str_columns_func(x, 'trafficSource.source', len, social_source_values))
    df['trafficSourceSocialRatio'] = column_z_score(sr.fillna(0))
    # se source ratio :
    sr = df_top_lvl.groupby('fullVisitorId').apply(lambda x: filter_str_columns_func(x, 'trafficSource.source', len, se_source_values))
    df['trafficSourceSeRatio'] = column_z_score(sr.fillna(0))
    # SOURCE_MEDIUM_FB_RATIO
    sr= df_top_lvl.groupby('fullVisitorId').apply(lambda x : filter_str_columns_func(x, 'trafficSource.medium', len, medium_fb_values))
    df['mediumSourceFbRatio'] = column_z_score(sr.fillna(0))
    # SOURCE_MEDIUM_CPC_RATIO
    sr = df_top_lvl.groupby('fullVisitorId').apply(lambda x: filter_str_columns_func(x, 'trafficSource.medium', len, 'cpc'))
    df['mediumSourceCpcRatio'] = column_z_score(sr.fillna(0))
    # SOURCE_MEDIUM_ORGANIC_RATIO
    sr = df_top_lvl.groupby('fullVisitorId').apply(lambda x: filter_in_columns_func(x, 'trafficSource.medium', len, organic_source_values))
    df['mediumSourceOrganicRatio'] = column_z_score(sr.fillna(0))
    # MEAN FOR TOTALS VIEWS
    df0 = df_by_group_by_single_column(df_top_lvl, 'fullVisitorId', 'totals.hits', 'mean')
    df['avg_hits'] = column_z_score(df0['totals.hits'].fillna(0))
    #
    df0 = df_by_group_by_single_column(df_top_lvl, 'fullVisitorId', 'totals.pageviews', 'mean')
    df['avg_page_views'] = column_z_score(df0['totals.pageviews'].fillna(0))
    #
    df0 = df_by_group_by_single_column(df_top_lvl, 'fullVisitorId', 'totals.timeOnSite', 'mean') 
    df['avg_time_on_site'] = column_z_score(df0['totals.timeOnSite'].fillna(0))
    # end of month ratio
    sr = group_by_start_or_end_of_month(df_top_lvl, 'visitStartTime', 'fullVisitorId', 'clientId', 'SM')
    df['end_of_month_ratio'] = column_z_score(sr.fillna(0))
    
    return df




# Utility functions
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
    return column_z_score(r, 'ratio')

'''
this method will be calculated in each column list of users
the vector will present the normal distribution of the values in each feature column 
'''

#  Convert to model functions

def convert_by_z(df, col_list) :
    def f(x, is_max):
        m = np.mean(x)
        s = np.std(x)
        h = m + 3*s
        l =  m-3*s
        r = h if is_max else l
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
    z = dft.apply(lambda x : x.corr(y))
    res =  z.map(lambda x : f(x))
    return res

