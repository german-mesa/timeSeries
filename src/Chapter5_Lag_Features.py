import os
from pandas import DataFrame, read_csv, concat


def main():
    # Read time serie to DataFrame
    series = read_csv(os.path.join(os.getcwd(), 'datasets', 'daily-minimum-temperatures.csv'),
                      header=0,
                      index_col=0,
                      parse_dates=[0])

    # Sliding window approach
    data = DataFrame(series.values)
    data = concat([data.shift(3), data.shift(2), data.shift(1), data], axis=1)
    data.columns = ['t-2', 't-1', 't', 't+1']

    # Some dataset info
    print(data.dtypes)
    print(data.describe())
    print(data.head(10))

    print('Number of records %d' % data.size)
    print(data.memory_usage(deep=True))


if __name__ == '__main__':
    main()
