#
# Time series using Pandas
# Example as described in https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
#
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def dataset_review(data):
    print(data.shape)
    print(data.dtypes)

    data['Year'] = data.index.year  # Adding year based on index
    data['Month'] = data.index.month  # Adding month based on index

    print(data.head(3))
    print(data.tail(3))
    print(data.sample(5, random_state=0))


def time_based_indexing(data):
    print(data.loc['2017-08-10'])  # Day
    print(data.loc['2014-01-20':'2014-01-22'])  # Slice of days
    print(data.loc['2012-02'])  # Partial slicing


def time_series_visualization(data):
    sns.set(rc={'figure.figsize': (11, 4)})

    # Full time series of Germany’s daily electricity consumption
    data['Consumption'].plot(linewidth=0.5)
    plt.show()

    # Subplots by generation type
    axes = data[['Consumption', 'Solar', 'Wind']].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9),
                                                       subplots=True)
    for ax in axes:
        ax.set_ylabel('Daily Totals (GWh)')
    plt.show()

    # Consumption in 2017 to visualize weekly seasonality. Note also decrease during winter holidays.
    ax = data.loc['2017', 'Consumption'].plot()
    ax.set_ylabel('Daily Consumption (GWh)')
    plt.show()

    # Consumption in Jan & Feb 2017 to visualize weekly seasonality
    ax = data.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')
    ax.set_ylabel('Daily Consumption (GWh)');
    plt.show()


def time_series_customizing(data):
    # Plots created directly in matplotlib
    fig, ax = plt.subplots()

    ax.plot(data.loc['2017-01':'2017-02', 'Consumption'], marker='o', linestyle='-')
    ax.set_ylabel('Daily Consumption (GWh)')
    ax.set_title('Jan-Feb 2017 Electricity Consumption')

    # Set x-axis major ticks to weekly interval, on Mondays
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))

    # Format x-tick labels as 3-letter month name and day number
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.show()


def time_series_seasonality(data):
    # Plots created directly in matplotlib
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for name, ax in zip(['Consumption', 'Solar', 'Wind'], axes):
        sns.boxplot(data=data, x='Month', y=name, ax=ax)

    ax.set_ylabel('GWh')
    ax.set_title(name)

    # Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')

    plt.show()


if __name__ == '__main__':
    # Read dataset from Open Power System data for Germany
    df = pd.read_csv("https://github.com/jenfly/opsd/raw/master/opsd_germany_daily.csv", index_col=0, parse_dates=True)

    # Review dataset
    dataset_review(df)

    # Indexing capabilities
    time_based_indexing(df)

    # Visualization capabilities
    time_series_visualization(df)

    # Customizing time series plots
    time_series_customizing(df)

    # Plotting seasonality
    time_series_seasonality(df)
