import os
from pandas import read_csv

import matplotlib.pyplot as plt


# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

def main():
    # Read time serie and squeeze it to a Series object
    series = read_csv(os.path.join(os.getcwd(), 'datasets', 'daily-total-female-births.csv'),
                      header=0,
                      index_col=0,
                      parse_dates=[0],
                      squeeze=True
                      )

    # Some dataset info
    print(series.dtypes)
    print(series.describe())
    print(series.head(10))
    print(series.tail(10))

    print('Number of records %d' % series.size)
    print('Memory usage %d bytes' % series.memory_usage(deep=True))

    # Filtering data

    # Basic plot
    series.plot()
    plt.show()


if __name__ == '__main__':
    main()
