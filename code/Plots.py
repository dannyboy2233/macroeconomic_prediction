"""
Defines matplotlib-based plotting functions to be used with specific data formats.

Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


import datetime as dt  # standard libraries

import matplotlib.pyplot as plt  # third-party libraries
import seaborn as sns; sns.set_style("darkgrid")
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters()
from sklearn.preprocessing import MinMaxScaler

from Utils import change_dir  # package libraries


def plot_from_date_indexed_df(df, cols, dic, plot_name, normalize=True, cutoff_lines=False):
    """
    Plots data of cols from date-indexed DataFrame against date.
    :param df: DataFrame containing data of cols
    :param cols: list of cols to plot
    :param dic: dict that maps key to variable name
    :param plot_name: name to save plot as --> plot_name.csv
    :param normalize: whether or not to normalize data
    :param cutoff_lines: whether or not to include cutoff lines at the four relevant dividing dates
    :return: nothing; saves the listed plot as .png in Economic Modelling/Figures
    """

    plt.clf()
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        plt.figure(num=None, figsize=(8, 3), dpi=1000)
        for c in cols:
            if normalize:
                scaler.fit(df[c].values.reshape(-1, 1))
                norm = scaler.transform(df[c].values.reshape(-1, 1))
                plt.plot_date(df['date'], norm, xdate=True, label=c, lw=1, marker='.', ms=1)
            else:
                plt.plot_date('date', c, data=df, xdate=True, label=c, lw=1, marker='.', ms=1)
        # if len(cols) == 2:
        #     plt.title("{} and {}: correlation = {}".format(cols[0], cols[1], corr(df, cols[0], cols[1])))
        else:
            plt.title(", ".join(cols))
        if cutoff_lines:
            xcoords = [dt.datetime(year=1977, month=6, day=1), dt.datetime(year=1986, month=4, day=1),
                       dt.datetime(year=1996, month=11, day=1), dt.datetime(year=2007, month=5, day=1)]
            for xc in xcoords:
                plt.axvline(x=xc, color='black', alpha=0.5, lw=1).set_dashes((2, 1))
        plt.legend()
        plt.yticks([0, 0.5, 1])
        change_dir("Figures")
        plt.savefig("{}.png".format(plot_name), format='png', dpi=1000, transparent=True)
    except KeyError as e:
        print("KeyError while creating plot:", e)
        pass
