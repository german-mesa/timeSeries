import os
from pandas import read_csv, DataFrame

import matplotlib.pyplot as plt


# https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html

def main():
    # Read time serie to DataFrame
    series = read_csv(os.path.join(os.getcwd(), 'datasets', 'daily-minimum-temperatures.csv'),
                      header=0,
                      index_col=0,
                      parse_dates=[0]
                      )

    # Dataframe with Date Time Features
    data = DataFrame()
    data['Date'] = series.index
    data['Year'] = series.index.year
    data['Month'] = series.index.month
    data['Day'] = series.index.day
    data['Temperature'] = series['Temp'].values

    data = data.set_index('Date')

    # Some dataset info
    print(data.dtypes)
    print(data.describe())
    print(data.head(10))
    print(data.tail(10))

    print('Number of records %d' % data.size)
    print(data.memory_usage(deep=True))

    # Basic plot
    ax = data['Temperature'].plot(linewidth=0.5)
    ax.set_ylabel('Temperature ºC')
    ax.set_title('Daily minimum temperature')
    plt.show()


if __name__ == '__main__':
    main()
