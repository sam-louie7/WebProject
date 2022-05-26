
traffic_source_values =  ['tenengroup.com','OrderReceived','mnn.custhelp.com','OrderPreview',
                        'catalog.tenengroup.com','Login','ThankYou','Shipping','(direct)','Confirmation']

social_source_values = 'facebook|instagram|criteo|Newsletter'

se_source_values = 'yahoo|bing|baidu|google'

medium_fb_values ='Facebook_|Instagram_'

organic_source_values = ['organic','referral','Email','Others']


top_level_drop_columns = ['index', 'date', 'hits', 'customDimensions', 'totals.index',
                          'geoNetwork.index', 'trafficSource.index', 'device.index', 'privacyInfo.index']
        

top_level_extract_columns = ['totals', 'trafficSource', 'device', 'geoNetwork', 'privacyInfo']


hits_drop_columns = ['hits.product', 'hits.promotion', 'hits.experiment', 'hits.publisher_infos',
                    'hits.customVariables', 'hits.customDimensions', 'hits.customMetrics',
                    'hits.hour', 'hits.minute']

products_drop_columns = ['hits.product.customMetrics', 'hits.product.customDimensions']
