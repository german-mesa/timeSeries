#
# Time series naive forecasting using Pandas
#
# Moving averages described in this article:
# https://towardsdatascience.com/moving-averages-in-python-16170e20f6c
#
# Electricity production and consumption are reported as daily totals in gigawatt-hours (GWh).
# The columns of the data file are:
#   Date — The date (yyyy-mm-dd format)
#   Consumption — Electricity consumption in GWh
#   Wind — Wind power production in GWh
#   Solar — Solar power production in GWh
#   Wind+Solar — Sum of wind and solar power production in GWh
#
import os
import pandas as pd
import numpy as np

from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def dataset_review(data):
    print(data.shape)
    print(data.dtypes)
    print(data.info())

    # data = data.set_index('Date')  # Creating DataFrame’s index as DatetimeIndex

    data['Year'] = data.index.year  # Adding year based on index
    data['Month'] = data.index.month  # Adding month based on index
    data['Weekday Name'] = data.index.day_name()

    print(data.head(3))
    print(data.tail(3))
    print(data.sample(5, random_state=0))


def time_series_visualization(data):
    # Patterns found in visualization:
    # - Electricity consumption is highest in winter, presumably due to electric heating and increased lighting usage,
    #   and lowest in summer.
    # - Electricity consumption appears to split into two clusters — one with oscillations centered roughly
    #   around 1400 GWh, and another with fewer and more scattered data points, centered roughly around 1150 GWh.
    #   We might guess that these clusters correspond with weekdays and weekends, and we will investigate this
    #   further shortly.
    # - Solar power production is highest in summer, when sunlight is most abundant, and lowest in winter.
    # - Wind power production is highest in winter, presumably due to stronger winds and more frequent storms,
    #   and lowest in summer.
    # - There appears to be a strong increasing trend in wind power production over the years.
    #
    sns.set(rc={'figure.figsize': (11, 4)})

    # Full time series of Germany’s daily electricity consumption
    data['Consumption'].plot(linewidth=0.5)
    plt.show()

    # Subplots by generation type
    axes = data[['Consumption', 'Solar', 'Wind']].plot(marker='.',
                                                       alpha=0.5,
                                                       linestyle='None',
                                                       figsize=(11, 9),
                                                       subplots=True)
    for ax in axes:
        ax.set_ylabel('Daily Totals (GWh)')
    plt.show()

    # Consumption in 2017 to visualize weekly seasonality. Note also decrease during winter holidays.
    colors = ['green', 'red', 'yellow', 'orange']
    parameters = ['SMA_10', 'CMA', 'EMA_0.1', 'EMA_0.3']

    ax = data.loc['2017', 'Consumption'].plot(linewidth=0.5)
    for param, color in zip(parameters, colors):
        data.loc['2017', param].plot(color=color, linewidth=3)

    ax.set_ylabel('Daily Consumption (GWh)')
    plt.show()

    # Consumption in Jan & Feb 2017 to visualize weekly seasonality
    ax = data.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')
    for param, color in zip(parameters, colors):
        data.loc['2017-01':'2017-02', param].plot(color=color, linewidth=3)

    ax.set_ylabel('Daily Consumption (GWh)')
    plt.show()


def mean_squared_error(value, forecast):
    # MSE measures the average magnitude of the error
    return keras.metrics.mean_squared_error(value, forecast).numpy()


def mean_absolute_error(value, forecast):
    # MAE measures the average magnitude of the errors in a set of predictions
    return keras.metrics.mean_absolute_error(value, forecast).numpy()


def moving_average_consumption(data, window):
    # Simple Moving Average - The simple moving average is the unweighted mean of the previous M data points.
    # Moving average smooth out the data, allowing us to properly visualize the trend direction.
    # The first rows of the returned series contain null values since rolling needs a minimum of M values  - value
    # specified in the window argument) to return the mean. You can change this behavior by modifying the argument
    # min_periods.The disadvantage of this method is that it could not smoothly decay old data and sometimes when
    # an outlier is added or discarded, the prediction will change a lot.
    data['SMA_10'] = data['Consumption'].rolling(window, min_periods=1).mean()
    print('MSE for SMA_10 {}'.format(mean_squared_error(data['Consumption'], data['SMA_10'])))
    print('MAE for SMA_10 {}'.format(mean_absolute_error(data['Consumption'], data['SMA_10'])))

    # Cumulative Moving Average - The Cumulative Moving Average is the unweighted mean of the previous values up
    # to the current time t, i.e. the window size becomes larger as the time passes when computing the cumulative
    # moving average. The cumulative moving average takes into account all the preceding values when calculating
    # the average. For this reason, they are a bad option to analyze trends, especially with long time series.
    data['CMA'] = data['Consumption'].expanding().mean()
    print('MSE for CMA {}'.format(mean_squared_error(data['Consumption'], data['CMA'])))
    print('MAE for CMA {}'.format(mean_absolute_error(data['Consumption'], data['CMA'])))

    # Exponential moving average - The exponential moving average is a widely used method to filter out noise and
    # identify trends. The weight of each element decreases progressively over time, meaning the exponential moving
    # average gives greater weight to recent data points. This is done under the idea that recent data is more
    # relevant than old data. Compared to the simple moving average, the exponential moving average reacts faster
    # to changes, since is more sensitive to recent movements.
    # A small weighting factor α results in a high degree of smoothing, while a larger value provides a quicker
    # response to recent changes.
    data['EMA_0.1'] = data['Consumption'].ewm(alpha=0.1, adjust=False).mean()
    print('MSE for EMA_0.1 {}'.format(mean_squared_error(data['Consumption'], data['EMA_0.1'])))
    print('MAE for EMA_0.1 {}'.format(mean_absolute_error(data['Consumption'], data['EMA_0.1'])))

    data['EMA_0.3'] = data['Consumption'].ewm(alpha=0.3, adjust=False).mean()
    print('MSE for EMA_0.3 {}'.format(mean_squared_error(data['Consumption'], data['EMA_0.3'])))
    print('MAE for EMA_0.3 {}'.format(mean_absolute_error(data['Consumption'], data['EMA_0.3'])))


def main():
    # Read dataset from Open Power System data for Germany. Uncomment the appropriate option.

    # File located into local repository
    data_file = os.path.join(os.getcwd(), 'datasets', 'opsd_germany_daily.csv')

    # File located into a remote URL
    # data_file = "https://github.com/jenfly/opsd/raw/master/opsd_germany_daily.csv"

    # Datetime index is created automatically when reading
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Review dataset
    dataset_review(df)

    # Adding moving averages
    moving_average_consumption(df, window=10)

    # Visualization capabilities
    time_series_visualization(df)


if __name__ == '__main__':
    main()
