#!/usr/bin/env python

"""
Performs all processes for project. Does everything related to FRED, then everything related to internal VMware
data, then does analysis. See individual scripts for more in-depth documentation of functions and processes.

Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


import warnings  # standard libraries

import pandas as pd  # third-party libraries

from Utils import change_dir, query_table, add_months  # package libraries
from FREDDataAggregation import get_data_dict, spline_interp_dict, aggregate_spline_interp_data, attach_dep_vars, upload
from VMWareInternalDataQuery import get_query_from_file, create
from MachineLearningAlgorithms import prep_for_training, augment_with_multivar_bootstrapped_boosted_tree_regression, \
                                      plot_all_dep_vars_vs_preds, five_fold_multivar_bstrapped_boosted_tree, \
                                      three_month_predictions
from Report import get_parameters, export_report, send_email


# temporarily define final parameters
MAX_YEARS = 60
MAX_DAYS = MAX_YEARS * 365

FIELDS_AND_DESCRIPTIONS = {
    # employment
    'UNRATE': 'Civilian Unemployment Rate',
    'ICSA': 'Initial Claims (unemployment)',
    'IC4WSA': '4-Week Moving Average of Initial Claims (unemployment)',

    # money
    'M1': 'M1 Money Stock',
    'M2': 'M2 Money Stock',
    'DTWEXM': 'Trade Weighted U.S. Dollar Index: Major Currencies, Goods',
    'DTWEXB': 'Trade Weighted U.S. Dollar Index: Broad, Goods',
    'DTWEXBGS': "Trade Weighted U.S. Dollar Index: Broad, Goods and Services",

    # markets
    'SP500': 'S&P 500',
    'DJIA': 'Dow Jones Industrial Average',
    'NASDAQCOM': 'NASDAQ Composite Index',

    # sentiment
    'UMCSENT': 'University of Michigan: Consumer Sentiment',
    'MICH': 'University of Michigan: Inflation Expectation',
    'CSCICP03USM665S': 'Consumer Opinion Surveys: OECED Confidence Indicators for US',
    'BSCICP03USM665S': """Business Tendency OECD Confidence Indicators
                          Survey for Manufacturing""",

    # housing
    'PERMIT': 'New Private Housing Units Authorized by Building Permits',
    'BOGZ1LM155035015A': 'Households; owner-occupied real estate at market level',
    'ASPUS': 'Average Sales Price of Houses Sold for the United States',
    'MSPUS': 'Median Sales Price of Houses Sold for the United States',
    'HSN1F': 'New One Family Houses Sold: United States',
    'ESALEUSQ176N': 'Housing Inventory Estimate: Vacant Housing Units for Sale in US',
    'HOUST': 'Housing Starts: Total: New Privately Owned Housing Units Started',
    'ETOTALUSQ176N': """Housing Inventory Estimate:
                    Total Housing Units for the United States""",

    # industry/manufacturing
    'AWHAEMAN': 'Average Weekly Hours of All Employees: Manufacturing',
    'LCEAMN01USM659S': 'Hourly Earnings: Manufacturing for the United States',
    'AWHMAN': """Average Weekly Hours of Production and Nonsupervisory
                 Employees: Manufacturing""",
    'AMTMNO': "Value of Manufacturers' New Orders for All Manufacturing Industries",
    'NEWORDER': "Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft",
    'ACDGNO': """Value of Manufacturers' New Orders for Consumer Goods:
                 Consumer Durable Goods Industries""",
    'DGORDER': "Manufacturers' New Orders: Durable Goods",
    'MANEMP': "All Employees: Manufacturing",
    'IPMAN': "Industrial Production: Manufacturing (NAICS)",
    'IPDCONGD': 'Industrial Production: Durable Consumer Goods',
    'A36SNO': """Value of Manufacturers' New Orders for Durable Goods Industries:
                 Transportation Equipment""",
    'AMDMUO': "Value of Manufacturers' Unfilled Orders for Durable Goods Industries",
    'AMDMUS': """Ratio of Manufacturers' Total Inventories to Unfilled Orders
                 for Durable Goods Industries""",
    'AMTMUO': "Value of Manufacturers' Unfilled Orders for All Manufacturing Industries",
    'TCU': 'Capacity Utilization: Total Industry',
    'INDPRO': 'Industrial Production Index',
    'MCUMFN': 'Capacity Utilization: Manufacturing (NAICS)',
    'BUSINV': 'Total Business Inventories',
    'TOTBUSMPCIMSA': 'Total Business Inventories',

    # consumption
    'PCDG': "Personal Consumption Expenditures: Durable Goods",
    'ACOGNO': "Value of Manufacturers' New Orders for Consumer Goods Industries",
    'T10YFF': '10-Year Treasury Constant Maturity Minus Federal Funds Rate',

    # business sales
    'TOTALSA': 'Total Vehicle Sales',
    'MARTSMPCSM44000USS': 'Advance Retail Sales: Retail (Excluding Food Services)',
    'RSXFS': 'Advance Retail Sales: Retail (Excluding Food Services)',
    'RETAILMPCSMSA': 'Retailers Sales',
    'ECOMSA': 'E-Commerce Retail Sales',
    'CPROFIT': 'Corporate Profits with with IVA and CCAdj',

    # price indices
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers: All Items',
    'CPALTT01USQ657N': 'Consumer Price Index: Total All Items for the United States',
    'CPALTT01USM657N': 'Consumer Price Index: Total All Items for the United States'
}
FIELDS = FIELDS_AND_DESCRIPTIONS.keys()
DEP_VARS = {'USREC': 'NBER based Recession Indicators for the United States', }
NUM_OFFSET_DAYS = 186  # for month purposes, should be in multiples of 31

VARIABLE_THRESHOLD = 20
TRAINING_START = pd.Timestamp(year=1971, month=10, day=1)
TRAINING_END = pd.Timestamp(year=2019, month=1, day=30)

RECIPIENTS = ['adye@vmware.com', 'janel@vmware.com', 'grayb@vmware.com', 'hoferm@vmware.com', 'allenmathew@vmware.com']

def execute_fred():
    """
    Executes commands required to prepare FRED data and export it to data warehouse.
    """

    change_dir("Table exports")

    di = get_data_dict(FIELDS, MAX_DAYS)
    spline_interp_dict(di)
    agg, src = aggregate_spline_interp_data(di)

    # agg = pd.read_csv("aggregate.csv").iloc[:, 1:]
    # src = pd.read_csv("sources.csv").iloc[:, 1:]

    df_dep_var_attach = attach_dep_vars(agg, DEP_VARS, NUM_OFFSET_DAYS, MAX_DAYS, monthly=True)

    upload({'dc_model_aggregate': agg,
            'dc_model_sources': src,
            'dc_model_merged': df_dep_var_attach})

    print("FRED PORTION COMPLETED.")


def execute_vmware_internal():
    """
    Executes commands required to prepare VMware internal data and save it in data warehouse.
    """

    query = get_query_from_file("daily_bookings_2010_onward.txt")
    create(query, "dc_model_vmware_sales_2010_onward")

    query = get_query_from_file("fred_vmware_combine.txt")
    create(query, "dc_model_fred_vmware_combined")

    print("VMWARE PORTION COMPLETED.")


def execute_ml():
    """
    Executes machine learning things.
    """

    dep_vars = ['USREC{}month'.format(i) for i in range(1, int(NUM_OFFSET_DAYS / 31) + 1) if i > 3]
    # dep_vars = ['USREC4month']
    df = query_table("dc_model_fred_vmware_combined")
    df_prepped, relevant_cols = prep_for_training(df, TRAINING_START, TRAINING_END)

    # BOOSTED REGRESSION TREE
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        five_fold_regressions = five_fold_multivar_bstrapped_boosted_tree(df_prepped, FIELDS, dep_vars, 3)
        df_complete = augment_with_multivar_bootstrapped_boosted_tree_regression(five_fold_regressions,
                                                                                 df_prepped, dep_vars)
    upload({'dc_model_complete': df_complete})
    change_dir("Table exports")
    # df_complete.to_csv('df_complete.csv')
    # plot_all_dep_vars_vs_preds(df_complete, dep_vars)

    predictions = three_month_predictions(df, five_fold_regressions, relevant_cols, NUM_OFFSET_DAYS)

    i = 1
    for pred in predictions:
        print('Probability in {} month(s):'.format(i), pred)
        print('Std Dev of probability in {} month(s):'.format(i), pred)
        i += 1

    print("MACHINE LEARNING PORTION COMPLETED.")


def execute_reporting():
    """
    Executes reporting methodology.
    """

    vals = get_parameters()
    export_report(vals[0], vals[1], vals[2], sources=query_table("dc_model_sources"))
    send_email(RECIPIENTS)

    print("REPORTING PORTION COMPLETED.")


def test():
    """
    For testing.
    """

    # dep_vars = ['USREC{}month'.format(i) for i in range(1, int(NUM_OFFSET_DAYS / 31) + 1)]
    # change_dir("Table exports")
    # df_complete = pd.read_csv("df_complete.csv").set_index('date_index', drop=True)
    # df_complete['date'] = [dt.datetime.strptime(date, '%Y-%m-%d') for date in df_complete['date']]
    # plot_all_dep_vars_vs_preds(df_complete, dep_vars)
    # change_dir("Table exports")
    # preds_table = pd.DataFrame(columns=['preds'], data=[1, 2, 3])
    # preds_table.to_csv('curr_preds_test.csv')
    # send_email(['adye@vmware.com'])


# execute_fred()
execute_vmware_internal()
execute_ml()
execute_reporting()
# test()
