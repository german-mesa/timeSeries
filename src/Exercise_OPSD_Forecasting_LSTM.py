#
# Time series naive forecasting using LSTM
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

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.utils as utils

import seaborn as sns
import matplotlib.pyplot as plt

SPLIT_TIME = 2500
WINDOW_SIZE = 64
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1000
RANDOM_SEED = 51


def dataset_review(data):
    # data['Year'] = data.index.year  # Adding year based on index
    # data['Month'] = data.index.month  # Adding month based on index
    # data['Weekday Name'] = data.index.day_name()

    data.info()

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

    titles = ['Germany’s daily electricity consumption',
              'Germany’s daily Solar production',
              'Germany’s daily Wind production']

    # Full time series of Germany’s daily electricity consumption
    ax = data['Consumption'].plot(linewidth=0.5)
    ax.set_title(titles[0])
    ax.set_ylabel('Daily Totals (GWh)')
    plt.show()

    # Subplots by generation type
    axes = data[['Consumption', 'Solar', 'Wind']].plot(marker='.',
                                                       alpha=0.5,
                                                       linestyle='None',
                                                       figsize=(11, 9),
                                                       subplots=True)

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_ylabel('Daily Totals (GWh)')

    plt.show()


def windowed_dataset(data, window_size, batch_size, shuffle_buffer):
    data = tf.expand_dims(data, axis=-1)
    data = np.asarray(data).astype('float32')

    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    ds = ds.batch(batch_size).prefetch(1)

    return ds


def forecasting_production(data):
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    data_train = data.loc['2006': '2017']
    data_validation = data.loc['2018': '2020']

    train_set = windowed_dataset(data_train['Consumption'], WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=5,
                strides=1,
                padding='causal',
                activation=tf.keras.activations.relu,
                input_shape=[None, 1]
            ),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dense(30, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 400)
        ])

    model.summary()

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),
        metrics=['mae']
    )

    history = model.fit(
        train_set,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
        ]
    )


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

    # Visualization capabilities
    time_series_visualization(df)

    # Forecasting production
    forecasting_production(df)


if __name__ == '__main__':
    main()
