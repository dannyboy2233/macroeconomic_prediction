"""
Performs preliminary regression analyses of FRED and VMware internal data.

Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


import time  # standard libraries
import os

import numpy as np  # third-party libraries
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from Utils import change_dir, is_dep_col  # package libraries


def corr(df, col1, col2):
    """
    Regresses col1 of df against col2 of df;
    Returns float between -1 and 1 (inclusive), and excludes rows for which there are not observations of
    one or both variables;
    If columns do not both exist in df, returns NaN.
    :param df: DataFrame containing values to be analyzed
    :param col1: First column to be regressed
    :param col2: Second column to be regressed
    :return: correlation coefficient (R**2) of the two columns
    """

    # print("Regressing %s against %s." % (col1, col2))
    # start_time = time.time()
    df_filtered = df[[col1, col2]].copy(deep=True)
    try:
        df_filtered = df_filtered.dropna()
        col1_vals = np.array(df_filtered[col1]).reshape(-1, 1)  # reshaping required for sklearn syntax
        col2_vals = np.array(df_filtered[col2])
        if len(col1_vals) != len(col2_vals):
            print("Filtering occurred improperly, please check process.")
            raise SystemExit
    except KeyError as e:
        print("Error while filtering df:", e)
        print("Verify that columns exist in your DataFrame.")
        return float('NaN')
    try:
        corr_val = LinearRegression().fit(col1_vals, col2_vals).score(col1_vals, col2_vals)
    except TypeError:
        print("One of columns is not of correct type. Ensure date column is not being used/all columns numeric.")
        return float('NaN')
    # end_time = time.time()
    # print("Regression of %s against %s completed in %s seconds." % (col1, col2, (end_time - start_time)))
    return corr_val


def regress_all(df, dep_keys):
    """
    Regresses each independent-variable column of df against each dependent-variable column of df;
    Uses dep_key to determine which columns are dependent-variable.
    :param df: DataFrame containing columns to be analyzed
    :param dep_keys: list of keys of dependent variables; all dependent-variable column names will start with
           one of these keys
    :return: DataFrame with three columns [col1, col2, corr]
    """

    if len(df.columns) < 2 or len(df) <= 0:
        print("DataFrame has too few rows/columns.")
        raise SystemExit
    print("Regressing all independent-variable columns of DataFrame against %s." % ','.join(dep_keys))
    start_time = time.time()
    cols = df.columns.values
    dep_cols = [s for s in cols if is_dep_col(s, dep_keys) and 'date' not in s]
    indep_cols = [s for s in cols if not is_dep_col(s, dep_keys) and 'date' not in s]
    df_regressed = pd.DataFrame(index=range(len(dep_cols) * len(indep_cols)),  # pre-fill df_regressed for efficiency
                                columns=['col1', 'col2', 'corr'])

    k = 0  # row indexer
    for i in indep_cols:
        for d in dep_cols:
            df_regressed.at[k, 'col1'] = i
            df_regressed.at[k, 'col2'] = d
            df_regressed.at[k, 'corr'] = corr(df, i, d)
            k += 1
    end_time = time.time()
    change_dir("/Users/dacohen/OneDrive - VMware, Inc/Economic Modelling/Table exports")
    df_regressed.to_csv("regressions.csv")
    print("%s regressions completed in %s seconds." % ((len(dep_cols) * len(indep_cols)), (end_time - start_time)))
    return df_regressed.sort_values(by='corr', ascending=False, na_position='last')


def get_label(c):
    """
    If label is first in a series, print it; if not, don't.
    :param c: column name
    :return: label to attach
    """

    # if c == 'GNPCA1' or c == 'GNPCA100':
    #     return c
    return ' '


def convert_to_regression_table(df, indep_vars, export_colored=False, fig_name=None):
    """
    Takes DataFrame w/ columns [col1, col2, corr] and converts to DataFrame with row for each col1 value and column
    for each col2 value s.t. df_converted.at[x, y] = corr(x, y).
    :param df: DataFrame to convert
    :param indep_vars: dict of independent variable keys and their descriptions
    :param export_colored: if True, will export df_converted as "heat-mapped" regression table (red is high
           positive, blue is high negative, grey is zero) to
    :param fig_name: name of figure to export, if export_colored == True
    :return: len(df['col1']) x len(df['col2']) DataFrame
    """

    if len(df) < 0 or set(df.columns.values) != {'col1', 'col2', 'corr'}:
        print("Incorrect column names or no rows.")
        raise SystemExit
    df_converted = pd.DataFrame(index=np.unique(df['col1']), columns=np.unique(df['col2']))
    for k in range(len(df)):  # for each row in original df
        try:
            df_converted.at[df.at[k, 'col1'], df.at[k, 'col2']] = np.float64(df.at[k, 'corr'])
        except KeyError as e:
            print("Error when accessing DataFrame:", e)
            return
    if export_colored:
        if fig_name is None:
            print("No name provided, setting to fig_test.")
            fig_name = 'fig_test'
        # labels = [list(df.iloc[k, :]) for k in range(len(df))]
        sns.set()
        fig, ax = plt.subplots()
        try:
            df_converted_np = [np.array(df_converted.loc[k].values, dtype=np.float64) for k in df_converted.index.values]
            ax = sns.heatmap(df_converted_np,  # df.to_numpy(),
                             xticklabels=[get_label(v) for v in df_converted.columns.values],
                             yticklabels=[v + ": " + indep_vars[v] for v in df_converted.index.values],
                             # annot=True,
                             cmap='bwr',
                             mask=pd.isnull(df_converted_np),
                             center=0,
                             vmin=-1,
                             vmax=1)
            curr_dir = os.getcwd()
            change_dir("/Users/dacohen/OneDrive - VMware, Inc/Economic Modelling/Figures")
            plt.savefig("{}.png".format(fig_name), format='png', dpi=1000, bbox_inches='tight')
            change_dir(curr_dir)
        except TypeError as e:
            print("Error creating heatmap:", e)
    return df_converted
