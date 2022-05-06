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



def init_top_lvl_df(df_list, numeric_cols, gb_col):
    df = pd.concat(df_list, ignore_index=True)
    # columns list to convert to numeric
    # df = convert_multiple_columns_to_numeric(df, cols=numeric_cols)
    return df


def init_hits_df(df_list, numeric_cols, time_col):
    df = pd.concat(df_list, ignore_index=True)
    #df = convert_multiple_columns_to_numeric(df, cols=numeric_cols)
    df = calc_delta_column(df, time_col, 'hits.deltaTimeMS')
    return df


def emotional_rational_hits_features(df_hits):
    # AVG_PRODUCTS_IN_SESSION
    df0 = df_group_by_mul_col(df_hits, 'numOfProduct', ['fullVisitorId', 'visitStartTime'], 'sum')
    df = df_by_group_by_single_column(df0, 'fullVisitorId', 'numOfProduct' , 'mean')
    df['hits.products_per_session'] = column_z_score(df, 'numOfProduct')
    df = df.drop('numOfProduct', axis=1)
    # AVG_VIEW_PRD
    df0 = df_filter_and_group_by_mul_col(df_hits, 'hits.eCommerceAction.action_type', [2], ['fullVisitorId', 'visitStartTime'], 'count')
    df0 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.eCommerceAction.action_type', 'mean')
    df['hits.avg_viewd_product'] = column_z_score(df0, 'hits.eCommerceAction.action_type')
    # AVG_AVG_TIMETOHIT
    df0 = df_group_by_mul_col(df_hits, 'hits.deltaTimeMS', ['fullVisitorId', 'visitStartTime'], 'mean')
    df0 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.deltaTimeMS', 'mean')
    df['hits.avg_time2hit'] = column_z_score(df0, 'hits.eCommerceAction.action_type')
    # AVG_TIME_ADD_PRD / RMV_PRD
    df0 = df_filter_and_agg_by_diff_col(df_hits, 'hits.eCommerceAction.action_type', [3], ['fullVisitorId', 'visitStartTime'], 'hits.time', 'mean')
    df0 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.time', 'mean')
    df['hits.avg_time_add_prod'] = column_z_score(df0, 'hits.time')
    df0 = df_filter_and_agg_by_diff_col(df_hits, 'hits.eCommerceAction.action_type', [4], ['fullVisitorId', 'visitStartTime'], 'hits.time', 'mean')
    df0 = df_by_group_by_single_column(df0, 'fullVisitorId', 'hits.time', 'mean')
    df['hits.avg_time_rmv_prod'] = column_z_score(df0, 'hits.time')
    # AVG_PAYMENT_PAGE
    df0 = df_group_by_mul_col(df_hits, 'hits.contentGroup.contentGroup1', ['fullVisitorId', 'visitStartTime'], 'count')
    df1 = df_filter_and_group_by_single_col(df0, 'hits.contentGroup.contentGroup1', 'Payment Page', 'fullVisitorId', 'mean')
    df['hits.avg_payment_page'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')
    # AVG_DROPDOWN_CLICK
    df1 = df_filter_and_group_by_single_col(df0, 'hits.contentGroup.contentGroup1', 'Dropdown Click', 'fullVisitorId', 'mean')
    df['hits.avg_dropdown_click'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')
    # AVG_SUPPORT_PAGE
    df1 = df_filter_and_group_by_single_col(df0, 'hits.contentGroup.contentGroup1', 'Support Page', 'fullVisitorId', 'mean')
    df['hits.avg_support_page'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')
    # # AVG_Article_PAGE
    df1 = df_filter_and_group_by_single_col(df0, 'hits.contentGroup.contentGroup1', 'Article Page', 'fullVisitorId', 'mean')
    df['hits.avg_article_page'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')
    # AVG_FEEDBACK_PAGE
    df1 = df_filter_and_group_by_single_col(df0, 'hits.contentGroup.contentGroup1', 'Feedback Page', 'fullVisitorId', 'mean')
    df['hits.avg_feedback_page'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')
    # AVG_SHOPPING_CART
    df1 = df_filter_and_group_by_single_col(df0, 'hits.contentGroup.contentGroup1', 'Shopping Cart', 'fullVisitorId', 'mean')
    df['hits.avg_shopping_cart'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')
    # AVG_DROP_FILTER_CLICK
    df1 = df_filter_and_group_by_single_col(df1, 'hits.contentGroup.contentGroup1', 'Drop Filter Click', 'fullVisitorId', 'mean')
    df['hits.avg_drop_filter_click'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')
    # AVG_CNT_SEARCH_KEYWORD
    df1 = df_filter_and_group_by_single_col(df1, 'hits.contentGroup.contentGroup1', 'Search Keyword', 'fullVisitorId', 'mean')
    df['hits.avg_search_keyword'] = column_z_score(df1, 'hits.contentGroup.contentGroup1')

    return df


def emotional_rational_toplvl_features(df_top_lvl):
    # tenen source ratio :
    df = df_top_lvl.groupby('fullVisitorId').apply(lambda x : filter_in_columns_func(x, ['trafficSource.source'], len, traffic_source_values).add_prefix('tenen.'))
    df['tenen.trafficSourceRatio'] = column_z_score(df, 'tenen.trafficSource.source')
    df = df.drop('tenen.trafficSource.source', axis=1)
    # social source ratio :
    df0 = df_top_lvl.groupby('fullVisitorId').apply(lambda x : filter_str_columns_func(x, ['trafficSource.source'], len, social_source_values).add_prefix('social.'))
    df['social.trafficSourceRatio'] = column_z_score(df0, 'social.trafficSource.source')
    # se source ratio :
    df0 = df_top_lvl.groupby('fullVisitorId').apply(lambda x: filter_str_columns_func(x, ['trafficSource.source'], len, se_source_values).add_prefix('se.'))
    df['se.trafficSourceRatio'] = column_z_score(df0, 'se.trafficSource.source')
    # SOURCE_MEDIUM_FB_RATIO
    df0 = df_top_lvl.groupby('fullVisitorId').apply(lambda x : filter_str_columns_func(x, ['trafficSource.medium'], len, medium_fb_values).add_prefix('fb.'))
    df['fb.mediumSourceRatio'] = column_z_score(df0, 'fb.trafficSource.medium')
    # SOURCE_MEDIUM_CPC_RATIO
    df0 = df_top_lvl.groupby('fullVisitorId').apply(lambda x: filter_str_columns_func(x, ['trafficSource.medium'], len, 'cpc').add_prefix('cpc.'))
    df['cpc.mediumSourceRatio'] = column_z_score(df0, 'cpc.trafficSource.medium')
    # SOURCE_MEDIUM_ORGANIC_RATIO
    df0 = df_top_lvl.groupby('fullVisitorId').apply(lambda x: filter_in_columns_func(x, ['trafficSource.medium'], len, organic_source_values).add_prefix('organic.'))
    df['organic.mediumSourceRatio'] = column_z_score(df0, 'organic.trafficSource.medium')


    df0 = df_by_group_by_single_column(df_top_lvl, 'fullVisitorId', 'totals.hits', 'mean')
    df['avg_hits'] = column_z_score(df0, 'totals.hits')
    df0 = df_by_group_by_single_column(df_top_lvl, 'fullVisitorId', 'totals.pageviews', 'mean')
    df['avg_page_views'] = column_z_score(df0, 'totals.pageviews')
    df0 = df_by_group_by_single_column(df_top_lvl, 'fullVisitorId', 'totals.timeOnSite', 'mean')
    df['avg_time_on_site'] = column_z_score(df0, 'totals.timeOnSite')
    # end of month ratio
    df0 = group_by_start_or_end_of_month(df_top_lvl, 'visitStartTime', 'fullVisitorId', 'clientId', 'SM')
    df['end_of_month_ratio'] = column_z_score(df0, 'clientId')
    # end of week ratio
    df0 = filter_by_weekends(df_top_lvl, 'visitStartTime')
    df_weekend = df_group_by_mul_col(df0, 'clientId', ['fullVisitorId', 'visitStartTime'], 'count')
    df_weekend = df_by_group_by_single_column(df_weekend, 'fullVisitorId', 'clientId', 'sum')
    df_all = df_group_by_mul_col(df_top_lvl, 'clientId', ['fullVisitorId', 'visitStartTime'], 'count')
    df_all = df_by_group_by_single_column(df_all, 'fullVisitorId', 'clientId', 'sum')
    df['end_of_week_ratio'] = ratio_on_df_columns(df_weekend['clientId'], df_all['clientId'] )
    return df




# Utility functions
def df_filter_and_group_by_mul_col(df, col, f_values, grp_by_cols, func):
    df0 = df[df[col].isin(f_values)].groupby(grp_by_cols).agg({col : [func]}).droplevel(1, axis=1).reset_index()
    return df0


def df_filter_and_group_by_single_col(df, col, f_value, grp_by_col, func):
    df0 = df[df[col] == f_value].groupby(grp_by_col).agg({col : [func]})
    return df0


def df_filter_and_agg_by_diff_col(df, filter_col, f_values, grp_by_cols, agg_col, func):
    df0 = df[df[filter_col].isin(f_values)].groupby(grp_by_cols).agg({agg_col : [func]}).droplevel(1, axis=1).reset_index()
    return df0



def df_group_by_mul_col(df, col, grp_by_cols, func):
    df0 = df.groupby(grp_by_cols).agg({col : [func]}).droplevel(1, axis=1).reset_index()
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
def convert_to_binary_col_by_mean(sr):
    # {1 : emotional,  -1 : rational}
    m = np.mean(sr)
    return  sr.map(lambda x : 1 if (x >= m) else -1)


def convert_to_reverse_binary_col_by_mean(sr):
    # {1 : emotional,  -1 : rational}
    m = np.mean(sr)
    return sr.map(lambda x : 1 if (x < m) else -1)


def corr_of_binary_columns(df, vec):
    def f(n):
        if n >= 0.3:  # high corr to emotional = 1
            return 1
        elif n <= -0.3: # low corr to emotional , rational = 2
            return 2
        else:  # no corr at all
            return 0
    dft = df.T
    idx = dft.index.to_list()
    y = pd.Series(vec, index=idx) # emotional vector
    z = dft.apply(lambda x : x.corr(y))
    res =  z.map(lambda x : f(x))
    return res


'''
hits.hitNumber,
hits.time,
hits.isInteraction,
hits.isEntrance,
hits.type,
hits.dataSource,
hits.uses_transient_token,
hits.page.pagePath,
hits.page.hostname,
hits.page.pageTitle,
hits.page.pagePathLevel1,
hits.page.pagePathLevel2,
hits.page.pagePathLevel3,
hits.page.pagePathLevel4,
hits.transaction.currencyCode,
hits.item.currencyCode,
hits.appInfo.screenName,
hits.appInfo.landingScreenName,
hits.appInfo.exitScreenName,
hits.appInfo.screenDepth,
hits.exceptionInfo.isFatal,
hits.eCommerceAction.action_type,
hits.eCommerceAction.step,
hits.social.socialNetwork,
hits.social.hasSocialSourceReferral,
hits.social.socialInteractionNetworkAction,
hits.contentGroup.contentGroup1,
hits.contentGroup.contentGroup2,
hits.contentGroup.contentGroup3,
hits.contentGroup.contentGroup4,
hits.contentGroup.contentGroup5,
hits.contentGroup.previousContentGroup1,
hits.contentGroup.previousContentGroup2,
hits.contentGroup.previousContentGroup3,
hits.contentGroup.previousContentGroup4,
hits.contentGroup.previousContentGroup5,
hits.contentGroup.contentGroupUniqueViews1,
hits.eventInfo.eventCategory,
hits.eventInfo.eventAction,
hits.eventInfo.eventLabel,
hits.latencyTracking.pageLoadSample,
hits.latencyTracking.pageLoadTime,
hits.latencyTracking.pageDownloadTime,
hits.latencyTracking.redirectionTime,
hits.latencyTracking.speedMetricsSample,
hits.latencyTracking.serverResponseTime,
hits.latencyTracking.domLatencyMetricsSample,
hits.latencyTracking.domInteractiveTime,
hits.latencyTracking.domContentLoadedTime,
hits.referer,
hits.transaction.transactionId,
hits.transaction.transactionRevenue,
hits.transaction.affiliation,
hits.transaction.localTransactionRevenue,
hits.transaction.transactionCoupon,
hits.item.transactionId,hits.isExit,
fullVisitorId,
visitStartTime,
hits.latencyTracking.domainLookupTime,
hits.latencyTracking.serverConnectionTime,
hits.transaction.transactionShipping,
hits.transaction.localTransactionShipping,
hits.page.searchKeyword,
hits.page.searchCategory,
numOfProducts
'''