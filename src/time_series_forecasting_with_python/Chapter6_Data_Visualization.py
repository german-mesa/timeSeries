import os

from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from pandas import concat
from pandas.plotting import lag_plot, autocorrelation_plot

import matplotlib.pyplot as plt

NUMBER_OF_ROWS = 5


# Check this for extra information:
# https://matplotlib.org/3.1.1/api/index.html#the-pyplot-api
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html

def main():
    # Read time serie to DataFrame
    series = read_csv(os.path.join(os.getcwd(), 'datasets', 'daily-minimum-temperatures.csv'),
                      header=0,
                      index_col=0,
                      parse_dates=True)

    print(series.head(NUMBER_OF_ROWS))

    # Line plot of the Minimum Daily Temperatures
    series.plot()
    plt.show()

    # Plot plot of the Minimum Daily Temperatures
    series.plot(style='k.')
    plt.show()

    # Stacked line plots of the Minimum Daily Temperatures
    years = DataFrame()
    for name, group in series.groupby(Grouper(freq="A")):
        years[name.year] = group['Temp'].values

    years.plot(subplots=True, legend=False)
    plt.show()

    # Histogram on the Minimum Daily Temperatures
    # The plotting function automatically selects the size of the bins based on the spread
    # of values in the data
    series.hist()
    plt.show()

    # Density Plot on the Minimum Daily Temperatures
    series.plot(kind='kde')
    plt.show()

    # Box and Whisker Plots
    # This plot draws a box around the 25th and 75th percentiles of the data that captures
    # the middle 50% of observations. A line is drawn at the 50th percentile (the median)
    # and whiskers are drawn above and below the box to summarize the general extents of the
    # observations. Dots are drawn for outliers outside the whiskers or extents of the data.

    # Yearly Box and Whisker Plots on the Minimum Daily Temperatures
    years.boxplot()
    plt.show()

    # Monthly Box and Whisker Plots on the Minimum Daily Temperatures - year 1990
    groups = series.loc['1990'].groupby(Grouper(freq="M"))
    months = concat([DataFrame(x[1].values) for x in groups], axis=1)
    months = DataFrame(months)
    months.columns = range(1, 13)

    months.boxplot()
    plt.show()

    # Heat Map
    # A matrix of numbers can be plotted as a surface, where the values in each cell of the matrix
    # are assigned a unique color. This is called a heatmap, as larger values can be drawn with warmer
    # colors (yellows and reds) and smaller values can be drawn with cooler colors (blues and greens)

    # Yearly Heat Map Plot of the Minimum Daily Temperatures - Transposed data
    plt.matshow(years.T, interpolation=None, aspect='auto')
    plt.show()

    # Monthly Heat Map Plot on the Minimum Daily Temperatures
    plt.matshow(months, interpolation=None, aspect='auto')
    plt.show()

    # Lag scatter
    # Time series modeling assumes a relationship between an observation and the previous observation.
    # Previous observations in a time series are called lags, with the observation at the previous time
    # step called lag1, the observation at two time steps ago lag=2, and so on.
    #
    # Pandas has a built-in function for exactly this called the lag plot. It plots the observation at time t on
    # the x-axis and the observation at the next time step (t+1) on the y-axis.
    # - If the points cluster along a diagonal line from the bottom-left to the top-right of the plot,
    #   it suggests a positive correlation relationship.
    # - If the points cluster along a diagonal line from the top-left to the bottom-right, it suggests
    #   a negative correlation relationship.
    #
    # More points tighter in to the diagonal line suggests a stronger relationship and more spread from the
    # line suggests a weaker relationship. A ball in the middle or a spread across the plot suggests a weak
    # or no relationship.

    # Lag scatter plot of the Minimum Daily Temperatures
    lag_plot(series)
    plt.show()

    # Autocorrelation Plot of the Minimum Daily Temperatures
    # The resulting plot shows lag along the x-axis and the correlation on the y-axis. Dotted lines are
    # provided that indicate any correlation values above those lines are statistically significant (meaningful).
    # We can see that for the Minimum Daily Temperatures dataset we see cycles of strong negative and positive
    # correlation. This captures the relationship of an observation with past observations in the same and
    # opposite seasons or times of year. Sine waves like those seen in this example are a strong sign of
    # seasonality in the dataset
    autocorrelation_plot(series)
    plt.show()


if __name__ == '__main__':
    main()
