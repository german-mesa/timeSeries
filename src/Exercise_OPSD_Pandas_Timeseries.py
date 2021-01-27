#
# Time series using Pandas
#
# Example as described in https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas/
#
# Electricity production and consumption are reported as daily totals in gigawatt-hours (GWh).
# The columns of the data file are:
#   Date — The date (yyyy-mm-dd format)
#   Consumption — Electricity consumption in GWh
#   Wind — Wind power production in GWh
#   Solar — Solar power production in GWh
#   Wind+Solar — Sum of wind and solar power production in GWh
#

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def dataset_review(data):
    print(data.shape)
    print(data.dtypes)

    # data = data.set_index('Date')  # Creating DataFrame’s index as DatetimeIndex

    data['Year'] = data.index.year  # Adding year based on index
    data['Month'] = data.index.month  # Adding month based on index
    data['Weekday Name'] = data.index.day_name()

    print(data.head(3))
    print(data.tail(3))
    print(data.sample(5, random_state=0))


def time_based_indexing(data):
    print(data.loc['2017-08-10'])  # Day
    print(data.loc['2014-01-20':'2014-01-22'])  # Slice of days
    print(data.loc['2012-02'])  # Partial slicing


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
    ax = data.loc['2017', 'Consumption'].plot()
    ax.set_ylabel('Daily Consumption (GWh)')
    plt.show()

    # Consumption in Jan & Feb 2017 to visualize weekly seasonality
    ax = data.loc['2017-01':'2017-02', 'Consumption'].plot(marker='o', linestyle='-')
    ax.set_ylabel('Daily Consumption (GWh)');
    plt.show()


def time_series_customizing(data):
    # To better visualize the weekly seasonality in electricity consumption in the plot above,
    # it would be nice to have vertical gridlines on a weekly time scale

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
    # Patterns found in visualization:
    # - Although electricity consumption is generally higher in winter and lower in summer, the median and lower
    #   two quartiles are lower in December and January compared to November and February, likely due to businesses
    #   being closed over the holidays. We saw this in the time series for the year 2017, and the box plot confirms
    #   that this is consistent pattern throughout the years.
    # - While solar and wind power production both exhibit a yearly seasonality, the wind power distributions have
    #   many more outliers, reflecting the effects of occasional extreme wind speeds associated with storms and
    #   other transient weather conditions.

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


def time_series_weekly_seasonality(data):
    # As expected, electricity consumption is significantly higher on weekdays than on weekends.
    # The low outliers on weekdays are presumably during holidays.

    sns.boxplot(data=data, x='Weekday Name', y='Consumption')

    plt.show()


def time_series_resampling_weekly(data):
    # Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

    # Resample to weekly frequency, aggregating with mean
    data_weekly_mean = data[data_columns].resample('W').mean()

    # Plot daily and weekly resampled time series together
    fig, ax = plt.subplots()

    ax.plot(data.loc['2017-01':'2017-06', 'Solar'],
            marker='.',
            linestyle='-',
            linewidth=0.5,
            label='Daily')

    ax.plot(data_weekly_mean.loc['2017-01':'2017-06', 'Solar'],
            marker='o',
            markersize=8,
            linestyle='-',
            label='Weekly Mean Resample')

    ax.set_ylabel('Solar Production (GWh)')
    ax.legend()

    plt.show()


def time_series_resampling_month(data):
    # At this monthly time scale, we can clearly see the yearly seasonality in each time series,
    # and it is also evident that electricity consumption has been fairly stable over time,
    # while wind power production has been growing steadily, with wind + solar power comprising an increasing
    # share of the electricity consumed

    # Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

    # Compute the monthly sums, setting the value to NaN for any month which has
    # fewer than 28 days of data
    data_monthly = data[data_columns].resample('M').sum(min_count=28)

    fig, ax = plt.subplots()
    ax.plot(data_monthly['Consumption'], color='black', label='Consumption')
    data_monthly[['Wind', 'Solar']].plot.area(ax=ax, linewidth=0)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.set_ylabel('Monthly Total (GWh)')

    plt.show()


def time_series_resampling_renewables_consumption_year(data):
    # Wind + solar production as a share of annual electricity consumption has been increasing
    # from about 15% in 2012 to about 27% in 2017

    # Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

    # Compute the annual sums, setting the value to NaN for any year which has
    # fewer than 360 days of data
    data_annual = data[data_columns].resample('A').sum(min_count=360)

    # The default index of the resampled DataFrame is the last day of each year,
    # ('2006-12-31', '2007-12-31', etc.) so to make life easier, set the index
    # to the year component
    data_annual = data_annual.set_index(data_annual.index.year)
    data_annual.index.name = 'Year'

    # Compute the ratio of Wind+Solar to Consumption
    data_annual['Wind+Solar/Consumption'] = data_annual['Wind+Solar'] / data_annual['Consumption']
    data_annual.tail(3)

    # Plot from 2012 onwards, because there is no solar production data in earlier years
    ax = data_annual.loc[2012:, 'Wind+Solar/Consumption'].plot.bar(color='C0')
    ax.set_ylabel('Fraction')
    ax.set_ylim(0, 0.3)
    ax.set_title('Wind + Solar Share of Annual Electricity Consumption')

    plt.xticks(rotation=0)

    plt.show()


def time_series_rolling_windows(data):
    # Unlike downsampling, where the time bins do not overlap and the output is at a lower frequency than the input,
    # rolling windows overlap and “roll” along at the same frequency as the data, so the transformed time series is
    # at the same frequency as the original time series

    # Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

    # Resample to weekly frequency, aggregating with mean
    data_weekly_mean = data[data_columns].resample('W').mean()

    # Compute the centered 7-day rolling mean - center=True argument to label each window at its midpoint
    data_rolling_7d = data[data_columns].rolling(7, center=True).mean()

    # Plot daily and weekly resampled time series together
    fig, ax = plt.subplots()

    ax.plot(data.loc['2017-01':'2017-06', 'Solar'],
            marker='.',
            linestyle='-',
            linewidth=0.5,
            label='Daily')

    ax.plot(data_weekly_mean.loc['2017-01':'2017-06', 'Solar'],
            marker='o',
            markersize=8,
            linestyle='-',
            label='Weekly Mean Resample')

    ax.plot(data_rolling_7d.loc['2017-01':'2017-06', 'Solar'],
            marker='.',
            linestyle='-',
            label='7-d Rolling Mean')

    ax.set_ylabel('Solar Production (GWh)')
    ax.legend()

    plt.show()


def time_series_trends_consumption(data):
    # Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

    # Compute the centered 7-day rolling mean - center=True argument to label each window at its midpoint
    data_rolling_7d = data[data_columns].rolling(7, center=True).mean()

    # The min_periods=360 argument accounts for a few isolated missing days in the
    # wind and solar production time series
    data_trend_365d = data[data_columns].rolling(window=365, center=True, min_periods=360).mean()

    # Plot daily, 7-day rolling mean, and 365-day rolling mean time series
    fig, ax = plt.subplots()

    ax.plot(data['Consumption'],
            marker='.',
            markersize=2,
            color='0.6',
            linestyle='None',
            label='Daily')

    ax.plot(data_rolling_7d.loc['2017-01':'2017-06', 'Solar'],
            marker='.',
            linestyle='-',
            label='7-d Rolling Mean')

    ax.plot(data_trend_365d['Consumption'],
            color='0.2',
            linewidth=3,
            label='Trend (365-d Rolling Mean)')

    # Set x-ticks to yearly interval and add legend and labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumption (GWh)')
    ax.set_title('Trends in Electricity Consumption')

    plt.show()


def time_series_trends_production(data):
    # Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
    data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']

    # The min_periods=360 argument accounts for a few isolated missing days in the
    # wind and solar production time series
    data_trend_365d = data[data_columns].rolling(window=365, center=True, min_periods=360).mean()

    # Plot 365-day rolling mean time series of wind and solar power
    fig, ax = plt.subplots()
    for nm in ['Wind', 'Solar', 'Wind+Solar']:
        ax.plot(data_trend_365d[nm], label=nm)
        # Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_ylim(0, 400)
        ax.legend()
        ax.set_ylabel('Production (GWh)')
        ax.set_title('Trends in Electricity Production (365-d Rolling Means)');

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

    # Plotting weekly seasonality
    time_series_weekly_seasonality(df)

    # Plotting average Electricity Consumption and renewables per week
    time_series_resampling_weekly(df)

    # Plotting Electricity Consumption and renewables per month
    time_series_resampling_month(df)

    # Plotting Wind + Solar Share of Annual Electricity Consumption per year
    time_series_resampling_renewables_consumption_year(df)

    # Plotting average and rolling mean Electricity Consumption and renewables per week
    time_series_rolling_windows(df)

    # Plotting Trends in Electricity Consumption
    time_series_trends_consumption(df)

    # Plotting Trends in Electricity Production
    time_series_trends_production(df)
