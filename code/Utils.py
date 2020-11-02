"""
Establishes utility functions for the Economic Model project that will be utilized throughout other scripts.

Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


import json  # standard libraries
import os

import sqlalchemy as sqlalc  # third-party libraries
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
from sqlalchemy.exc import DBAPIError, SQLAlchemyError


def change_dir(f):
    """
    Changes directory to folder f within /Users/dacohen/OneDrive - VMware, Inc/Economic Modelling if aws=False,
    and to folder d_modified within /home/ec2-user if aws=True.
    :param f: desired folder to change to
    """

    alans_machine = True

    try:
        if "/" in f:  # if f is actually a directory
            os.chdir(f)
        else:  # if f is indeed a folder
            if not alans_machine:
                d = "/Users/dacohen/OneDrive - VMware, Inc/Economic Modelling/{}".format(f)
            else:
                d = "/Users/aldye/Applications/Danny Intern Model For Alan/{}".format(f)
            os.chdir(d)
    except OSError:
        print("Cannot change working directory.")


def get_credentials(file_name):
    """
    Gets data warehouse credentials from file. Ensures user, password, etc. are not visible within code itself;
    Must use file in following format: {"host": _, "port": _, "db": _, "user": _, "pw": _};
    Within function, switches to Credentials directory, then back to whichever directory the user was in before
    :param file_name: extension of .txt file containing credentials
    :return: dict mapping field to value
    """

    print("Collecting credentials.")
    curr_dir = os.getcwd()
    try:
        change_dir("Credentials")
    except OSError:
        print("Cannot change to Credentials directory.")
        raise SystemExit
    try:
        with open(file_name) as f:
            change_dir(curr_dir)
            return json.load(f)
    except OSError as e:
        print("Error downloading credentials:", e)
        print("Verify file integrity/formatting and try again.")
        raise SystemExit


def quarter(month):
    """
    Helper function to convert month to quarter.
    :param month: a month
    :return: the quarter the specified month is in
    """

    return {
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 3,
        10: 4, 11: 4, 12: 4
    }[month]


def query_table(tbl_name):
    """
    Queries table tbl_name from VMware data warehouse; runs "SELECT * FROM tbl_name".
    :param tbl_name: name of table to query
    :return: DataFrame containing results of query
    """

    engine = get_sqlalchemy_engine()
    try:
        return pd.read_sql_query("SELECT * FROM {};".format(tbl_name), con=engine)
    except SQLAlchemyError or DBAPIError as e:
        print("Error in query:", e)
        print("Verify queried table exists in data warehouse.")
        return


def get_sqlalchemy_engine():
    """
    Gets SQLalchemy engine with VMware data warehouse credentials.
    :return: SQLalchemy engine
    """

    creds = get_credentials("creds.txt")
    connection_string = "postgresql+psycopg2://{u}:{pw}@{h}:{po}/{db}".format(u=creds['user'],
                                                                   pw=creds['pw'],
                                                                   h=creds['host'],
                                                                   po=creds['port'],
                                                                   db=creds['db'])
    return sqlalc.create_engine(connection_string, pool_pre_ping=True)


def is_dep_col(col, dep_keys):
    """
    Helper function that determines whether or not column is dependent.
    :param col: column name in question
    :param dep_keys: list of dependent keys
    :return: True if column is dependent, false if not
    """

    if len(col) > 3:
        if col[:3] == 'vmw':
            return True
    for key in dep_keys:
        if len(col) >= len(key):
            if col[:len(key)] == key:
                return True
    return False


def days_since_start_date(date, start):
    """
    Helper function that returns number of days in supplied date since start;
    Used for spline interpolation
    :param date: date from which to calculate days
    :param start: start date
    :return: integer value of days between the two dates
    """

    return int((date - start).days)


def is_date(val):
    """
    Determines if val is a date (Python Datetime/Date, Numpy Datetime64, or Pandas Timestamp).
    :param val: value whose date-ness is to be determined
    :return: True if val is a date, False if it is not
    """

    if isinstance(val, int) or isinstance(val, float):
        return False
    try:
        pd.to_datetime(val, errors='raise')
        return True
    except ValueError:
        return False


def is_binary(k):
    """
    Returns whether or not key k is a binary variable.
    :param k: key to be analyzed
    :return: True if k is binary (takes on only 1 or 0), False otherwise
    """

    return k in ['USREC']


def subtract_months(l, n):
    """
    Subtracts n months DESTRUCTIVELY from each date in list l.
    :param l: list of dates
    :param n: number of months to subtract
    :return: new list with same number of elements, and each date having one month subtracted.
    """

    return [d + relativedelta(months=(-1*n)) for d in l]


def add_months(d, n):
    """
    Adds n months to date d.
    :param d: date
    :param n: number of months to add
    :return: new date with one month added
    """

    return d + relativedelta(months=n)


def min_max_normalized(data):
    """
    Performs min-max normalization of inputted data.
    :param data: list of values to normalize.
    :return: Lit of normalized values
    """

    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


def max_day(month):
    """
    Returns integer value of max day of given month.
    :param month: month
    :return: max day of that month
    """

    if month in [9, 4, 6, 11]:
        return 30
    elif month == 2:
        return 28
    else:
        return 31
