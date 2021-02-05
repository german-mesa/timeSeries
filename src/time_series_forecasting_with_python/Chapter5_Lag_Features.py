import os
from pandas import DataFrame, read_csv, concat

WINDOW_WITH = 3
NUMBER_OF_ROWS = 5


def main():
    # Read time serie to DataFrame
    series = read_csv(os.path.join(os.getcwd(), 'datasets', 'daily-minimum-temperatures.csv'),
                      header=0,
                      index_col=0,
                      parse_dates=[0])

    # Original data
    print(series.head(NUMBER_OF_ROWS))

    # Sliding window approach
    # The simplest approach is to predict the value at the next time (t+1) given the value
    # at the current time (t)
    data = DataFrame(series.values)
    data = concat([data.shift(1), data], axis=1)
    data.columns = ['Shifted', 'Original']
    print(data.head(NUMBER_OF_ROWS))

    # Now we include the last 3 observed values to predict the value at the next time step.
    # we must discard the first few rows that do not have enough data to train a supervised model.
    data = DataFrame(series.values)
    data = concat([data.shift(3), data.shift(2), data.shift(1), data], axis=1)
    data.columns = ['t-2', 't-1', 't', 't+1']
    print(data.head(NUMBER_OF_ROWS))

    # Rolling Window Statistics
    # We can perform statistical functions on the window of values collected for each time step,
    # such as calculating the mean. First, the series must be shifted. Then the rolling dataset
    # can be created and the mean values calculated on each window.
    #
    # For our particular case:
    # In this case, the window width of 3 means we must shift the series forward by 2 time steps.
    # - This makes the first two rows NaN.
    #   0   NaN
    #   1   NaN
    #   2  20.7
    #   3  17.9
    #   4  18.8
    #
    # - Next, we need to calculate the window statistics with 3 values per window. It takes 3
    #   rows before we even have enough data from the series in the window to start calculating
    #   statistics.
    #
    #   Window Values
    #   1,     NaN
    #   2,     NaN, NaN
    #   3,     NaN, NaN, 20.7
    #   4,     NaN, 20.7, 17.9
    #   5,     20.7, 17.9, 18.8
    #
    #   Therefore the values in the first 5 windows are:
    #       min       mean   max   t+1
    #   0   NaN        NaN   NaN  20.7
    #   1   NaN        NaN   NaN  17.9
    #   2   NaN        NaN   NaN  18.8
    #   3   NaN        NaN   NaN  14.6
    #   4  17.9  19.133333  20.7  15.8
    #
    data = DataFrame(series.values)
    shifted = data.shift(WINDOW_WITH - 1)
    print(shifted.head(NUMBER_OF_ROWS))

    window = shifted.rolling(window=WINDOW_WITH)
    data = concat([
        window.min(),
        window.mean(),
        window.max(),
        data
    ], axis=1)
    data.columns = ['min', 'mean', 'max', 't+1']
    print(data.head(NUMBER_OF_ROWS))


if __name__ == '__main__':
    main()
