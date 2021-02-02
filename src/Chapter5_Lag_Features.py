import os
from pandas import DataFrame, read_csv, concat


def main():
    # Read time serie to DataFrame
    series = read_csv(os.path.join(os.getcwd(), 'datasets', 'daily-minimum-temperatures.csv'),
                      header=0,
                      index_col=0,
                      parse_dates=[0])

    # Sliding window approach
    # The simplest approach is to predict the value at the next time (t+1) given the value
    # at the current time (t)
    data = DataFrame(series.values)
    data = concat([data.shift(1), data], axis=1)
    data.columns = ['Shifted', 'Original']
    print(data.head(10))

    # Now we include the last 3 observed values to predict the value at the next time step
    data = DataFrame(series.values)
    data = concat([data.shift(3), data.shift(2), data.shift(1), data], axis=1)
    data.columns = ['t-2', 't-1', 't', 't+1']
    print(data.head(10))


if __name__ == '__main__':
    main()
