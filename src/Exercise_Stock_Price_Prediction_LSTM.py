import math
import pandas_datareader as web

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler


def dataset_review(data):
    data.info()

    print(data.head(3))
    print(data.tail(3))
    print(data.sample(5, random_state=0))


def time_series_visualization(data):
    # Plot all together
    plt.plot(data['High'], linestyle='dotted', color='skyblue', linewidth=0.5)
    plt.plot(data['Close'], linestyle='solid', color='skyblue', linewidth=1)
    plt.plot(data['Low'], linestyle='dotted', linewidth=0.5)
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.legend()
    plt.show()

    # Subplots by price type
    axes = data[['High', 'Low', 'Open', 'Close', 'Volume']].plot(marker='.',
                                                                 alpha=0.5,
                                                                 linestyle='None',
                                                                 figsize=(11, 9),
                                                                 subplots=True)

    for ax in axes:
        ax.set_xlabel('Date')
        ax.set_ylabel('Price USD')

    plt.show()


def main():
    # Read stock prices dataset from Yahoo
    df = web.DataReader('MSFT', data_source='yahoo', start='2012-01')

    # Review dataset
    dataset_review(df)

    # Visualization capabilities
    time_series_visualization(df)


if __name__ == '__main__':
    main()
