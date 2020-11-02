"""
Queries relevant data from Federal Reserve Economic Data (FRED) database, fuzzily interpolates to daily values,
and aggregates into master table; then joins with table of dependent-variable observations extended NUM_OFFSET_DAYS
into the future. Script used exclusively for data generation, NO ANALYSIS.

Utilizes "fred" package, a third-party API for accessing the FRED database. In the future, this API may become
deprecated; in this case, either use the default FRED API and read data in as .json or .xml and convert, or
make an attempt to find another third-party API that is maintained.

NB: FRED API use requires free personal API key. I have specified my key here as FRED_API_KEY; if this key
ever becomes deprecated, please visit https://research.stlouisfed.org/useraccount/apikeys and request
your own key.

Primary exports:
    sse_ccmi.dc_model_aggregate: table with date column and column for each observed field
    sse_ccmi.dc_model_merged: aggregate merged with offset table of each dependent variable
    sse_ccmi.dc_model_sources: table containing each key and its listed source in the FRED database or NA

Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


import io  # standard libraries
from urllib.error import HTTPError
import warnings
import datetime as dt
from functools import reduce
import time

from fred import Fred  # third-party libraries
import pandas as pd
# import psycopg2
# import sqlalchemy as sqlalc
from sqlalchemy.exc import NotSupportedError
from scipy import interpolate
import numpy as np

from Utils import change_dir, get_credentials, quarter, get_sqlalchemy_engine, \
                  days_since_start_date, is_date, is_binary, subtract_months  # package libraries

# set up access to FRED API
FRED_API_KEY = '71cc92929a13c976ab205e3389701e9f'
FRED = Fred(api_key=FRED_API_KEY, response_type='dict')


def get_data(k, max_days):
    """
    Queries FRED database to find data corresponding to given field name.
    :param k: string that corresponds to FRED data series
    :param max_days: max number of days to go back in data
    :return: Pandas DataFrame with columns: year, quarter, month, day, value, freq, and source; and index=date
    """

    print("Querying FRED for %s data." % k)
    start_time = time.time()
    warnings.filterwarnings("error")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        try:
            da = FRED.series.observations(k)
            # da = FRED.get_series(k)
            # print(da)
            # info = FRED.get_series_info(k)
            # print(info)
            df = pd.DataFrame(da)

            try:
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                df['quarter'] = df['month'].apply(quarter)
            except KeyError:
                print('Date field not present in' + k + ' dataset.')
                df['year'] = None
                df['month'] = None
                df['day'] = None
                df['quarter'] = None
            try:
                df['freq'] = FRED.series.details(k)[0]['frequency']
            except KeyError:
                print('Frequency not found as field with key ' + k + '. Replacing with NA.')
                df['freq'] = None
            try:
                df['source'] = FRED.series.release(k)[0]['link']
            except KeyError:
                print('Source not found as field with key ' + k + '. Replacing with NA.')
                df['source'] = None
            min_date = dt.date.today() - dt.timedelta(days=max_days)
            df = df[df['date'] >= min_date]
            df.reset_index(drop=True, inplace=True)
            df = df[['date', 'year', 'quarter', 'month', 'day', 'value', 'freq', 'source']]
            df.set_index('date', inplace=True, drop=False)
            df.index.names = ['date_index']
            end_time = time.time()
            print("Successfully queried data for %s in %s seconds" % (k, (end_time - start_time)))
            return df
        except HTTPError as e:
            print("HTTP Error in key %s:" % k, e)
            time.sleep(5)
            return get_data(k, max_days)


def get_data_dict(l, max_days):
    """
    Creates dictionary linking field name to corresponding DataFrame.
    :param l: list of strings
    :param max_days: max number of days to go back for data
    :return: dict of names & DataFrames
    """

    if len(l) <= 0:
        print("List must have length greater than zero.")
        raise SystemExit

    df_dict, num_queries, n = {}, 0, 20
    for name in l:
        if num_queries > 0 and num_queries % 10 == 0:  # velocity limiting for the server
            print("Pausing for %s seconds..." % n)
            time.sleep(n)
        data_temp = get_data(name, max_days)
        if data_temp is not None and len(data_temp) > 0:
            df_dict[name] = data_temp
        num_queries += 1
    return df_dict


def spline_interp_df(k, df):
    """
    Takes DataFrame containing time series data, performs spline interpolation;
    If fatal error, returns None and data series is excluded from final dict.
    :param k: key associated with input DataFrame
    :param df: DataFrame containing time series
    :return: new DataFrame with time series interpolated to daily values; index='date', columns=[_key_, source]
    """

    start_time = time.time()
    print("Spline-interpolating %s data." % k)
    try:
        min_date, max_date = min(df['date']), max(df['date'])
        num_days_diff = (max_date - min_date).days
    except KeyError as e:
        print("Error while selecting date column:", e)
        print("Ensure input DataFrame contains date column.")
        return
    daily_date_vals = pd.date_range(min_date, periods=num_days_diff, freq='D').tolist()
    df_interp = pd.DataFrame(index=daily_date_vals)
    df_interp['date'] = daily_date_vals
    df_interp['source'] = df.iloc[0]['source']

    daily = False
    try:
        daily = df['freq'].iloc[0] == 'Daily'
        if daily:  # if data are daily, check for NA values
            if df.isnull().any()['value']:
                print("%s data are daily and there are NA values; excluding rows w/ NA values." % k)
                df.dropna(axis='index', subset=['value'], thresh=1, inplace=True)
            else:
                print("%s data are daily and there are no NA values; no interpolation needed.")
                df = df[['date', 'value', 'source']].rename(columns={'value': k})
                return df
    except IndexError as e:
        print("Error checking frequency of", k, "(likely has no data):", e)
        return
    except KeyError as e:
        print("Error checking frequency of", k, ":", e)
        pass

    x = np.array([(d - min_date).days for d in df['date']])
    y = np.array(df['value'])
    xnew = np.arange(0, num_days_diff)
    if is_binary(k):
        print("Key %s is binary, filling in instead of interpolating." % k)
        interpolator = interpolate.interp1d(x, y, kind='zero')
        ynew = interpolator(xnew)
    elif daily:
        interpolator = interpolate.interp1d(x, y, kind='linear')
        ynew = interpolator(xnew)
    else:
        tck = interpolate.splrep(x, y, s=0)
        ynew = interpolate.splev(xnew, tck, der=0)
    df_interp[k] = ynew
    end_time = time.time()
    print("Spline-interpolated %s data in %s seconds" % (k, (end_time - start_time)))
    return df_interp[['date', k, 'source']]


def spline_interp_dict(dic):
    """
    Takes set of time series; if series is non-daily, spline interpolates IN PLACE;
    If spline interpolation not possible, removes key from dict.
    :param dic: dict that maps key (e.g. 'GNPCA') to DataFrame containing that key's data
    :return: dict that maps keys to fuzzily interpolated DataFrames
    """

    for key, df in dic.items():
        new_df = spline_interp_df(key, df)
        try:
            dic[key] = new_df if new_df is not None else dic.pop(key, None)
        except KeyError:
            pass
    return dic


def aggregate_spline_interp_data(dic):
    """
    Returns aggregate table of data post-spline-interpolation. Assumes all data are daily.
    :param dic: dict that maps keys to DataFrames; cols: _key_, source and index: date
    :return: aggregate DataFrame and source DataFrame
    """

    print("Aggregating spline-interpolated data.")
    start_time = time.time()
    sources_row_list = []
    for k, df in dic.items():
        try:
            new_dict = {'key': k, 'source': df['source'].iloc[0]}
            sources_row_list.append(new_dict)
            dic[k] = df[['date', k]]  # re-adds DataFrame without 'source' column so it isn't included in aggregate
        except KeyError:
            dic[k] = df[['date', k]]
            print("Source column not present in table of key" + k + ". Please add and rerun.")
        except IndexError as e:
            print("Error extracting source from", k, "(likely no data):", e)
            pass
    sources = pd.DataFrame(sources_row_list)

    df_list = [df for df in dic.values() if len(df) > 0]
    try:
        df_agg = reduce(lambda x, y: pd.merge(x, y, on='date', left_index=False, how='outer'), df_list)
    except ValueError as e:
        print("Issue merging:", e)
        raise SystemExit
    # df_agg.to_csv("aggregate.csv")
    # sources.to_csv("sources.csv")
    end_time = time.time()
    print("Aggregated spline-interpolated data in %s seconds" % (end_time - start_time))
    return df_agg, sources


def offset_dependent_var(n, k, df, monthly=False):
    """
    Takes dependent variable time-series and offsets by 1-to-n days.
    :param n: maximum number of days to offset data (suggest no higher than 365)
    :param k: key of variable that is being offset
    :param df: DataFrame w/ _key_ column, date index
    :param monthly: if True, will offset monthly n // 31 times; otherwise will offset daily
    :return: DataFrame with n+2 columns: date, _key_, _key_ @ + 1 day, _key_ @ + 2 days, ..., _key_ @ + n days
    """

    if n < 0:
        print("n cannot be less than zero. Exiting.")
        raise SystemExit
    print("Offsetting %s data." % k)
    start_time = time.time()
    try:
        df.drop(columns=['source'], inplace=True)
    except KeyError:
        pass
    try:
        if monthly:
            if 'date' not in df.columns.values:
                df['date'] = [dt.datetime.utcfromtimestamp(e.tolist() / 1e9) for e in df.index.values]
            df_copy = df.copy(deep=True)
            num_offsets = n // 31
            for i in range(1, num_offsets + 1):
                col_str = k + str(i) + "month"
                df_shifted = df.copy(deep=True)
                # df_shifted.set_index(subtract_months(df_shifted.index.values, i), inplace=True)
                df_shifted['date'] = subtract_months(df['date'], i)
                df_shifted.rename(columns={k: col_str}, inplace=True)
                df_copy = pd.merge(df_copy, df_shifted, on='date', how='left')
            df_copy.drop_duplicates(inplace=True)
            df_copy.fillna(method='ffill', inplace=True)
        else:
            df_copy = df.copy(deep=True)
            for i in range(1, n + 1):
                col_str = k + str(i)
                df_copy[col_str] = df_copy[k].shift(-1 * i)
            num_offsets = n
    except KeyError as e:
        print("Error in offsetting:", e)
        print("Exiting; please ensure key %s present in DataFrame." % k)
        raise SystemExit
    # df_copy['date'] = [dt.datetime.utcfromtimestamp(e.tolist()/1e9) for e in df_copy.index.values]
    df_copy.set_index('date', drop=False, inplace=True)
    df_copy.index.names = ['date_index']
    # df_copy.to_csv(k + "_offset_" + str(num_offsets) + "x.csv")
    end_time = time.time()
    print("Offset data", k, "on", str(num_offsets), "units in %s seconds" % (end_time - start_time))
    return df_copy


def attach_dep_vars(df, d_v, num_offset_days, max_days, monthly=False):
    """
    Takes DataFrame of dates/fields and attaches offset dependent variable observations;
    :param df: DataFrame containing date column and one column for each field
    :param d_v: dict of keys of dependent variables being used for dataset
    :param num_offset_days: number of days to offset dependent variable
    :param max_days: number of days to go back in data
    :param monthly: if True, will offset monthly to n x 31 days; otherwise will offset daily
    :return: copy of df with offset dependent variables joined by date
    """

    df_copy = df.copy(deep=True)
    d_v_dict = get_data_dict(d_v, max_days)
    d_v_dict_interp = spline_interp_dict(d_v_dict)
    offset_dfs = []
    for k in d_v.keys():
        print("Appending %s to list of offset DataFrames." % k)
        start_time = time.time()
        offset_dfs.append(offset_dependent_var(num_offset_days, k, d_v_dict_interp[k], monthly))
        end_time = time.time()
        print("Appended", k, "to list of offset DataFrames in %s seconds." % (end_time - start_time))
    # df_copy['date'] = pd.to_datetime(df_copy.loc[:, 'date'], errors='coerce')
    # df_copy = df_copy[df_copy['date'].notnull()]
    for elem in offset_dfs:
        # elem['date'] = pd.to_datetime(elem.loc[:, 'date'], errors='coerce')
        # elem = elem[elem['date'].notnull()]
        try:
            df_copy = pd.merge(df_copy, elem, on='date', how='outer')
        except ValueError as e:
            print("Issue merging:", e)
            return
    # df_copy.to_csv("df_merged.csv")
    return df_copy


def upload_table(connection, cursor, engine, df, tbl_name):
    """
    Helper function for upload() that takes established connection and uploads the DataFrame with given name.
    Source: https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table
    :param connection: connection used to communicate with PostgreSQL database
    :param cursor: PostgreSQL connection to use for uploading
    :param engine: SQLalchemy engine to be used with pd.to_sql()
    :param df: DataFrame to be uploaded into Data Warehouse
    :param tbl_name: desired name for table, with dc_model_ as prefix
    """

    print("Uploading table %s." % tbl_name)
    start_time = time.time()

    if df is None:
        print("DataFrame provided is NoneType object. Nothing to upload.")
        return

    df.head(0).to_sql(tbl_name, engine, if_exists='replace', index=False)
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    cursor.copy_from(output, tbl_name, null="")  # null values become ''
    connection.commit()

    end_time = time.time()
    print("Successfully uploaded table", tbl_name, "in %s seconds" % (end_time - start_time))


def upload(df_dic):
    """
    Uploads DataFrame to VMware Data Warehouse as dc_model_*name* for each name and DataFrame in df_dic.
    :param df_dic: dict that maps desired name to DataFrame containing information
    """

    try:
        engine = get_sqlalchemy_engine()
        connection = engine.raw_connection()
        cursor = connection.cursor()
        print("Connected to PostgreSQL database.")
        for name, df in df_dic.items():
            upload_table(connection, cursor, engine, df, name)

        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed.")
    except NotSupportedError as e:
        print("Error while connecting to database:", e)
        raise SystemExit
