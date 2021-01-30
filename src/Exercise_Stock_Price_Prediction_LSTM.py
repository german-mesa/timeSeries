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
    plt.plot(data['Close'], linestyle='solid', color='skyblue', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Close Price USD')

    plt.show()


def build_model_stock_price_prediction(data):
    pass


def main():
    # Read daily stock prices dataset from Yahoo
    df = web.DataReader('MSFT', data_source='yahoo', start='2012-01')

    # Review dataset
    dataset_review(df)

    # Time series visualization
    time_series_visualization(df)

    # Built model for stock price prediction
    model = build_model_stock_price_prediction(df)


if __name__ == '__main__':
    main()
