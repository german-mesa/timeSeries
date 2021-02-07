import os
import math
import pandas_datareader as web

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

SPLIT_TIME = 2500
WINDOW_SIZE = 64
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1000
RANDOM_SEED = 51


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


def build_model_stock_price_prediction(data):
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'MSFT-{epoch:04d}.ckpt')

    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    data_train = data.loc['2012': '2019']
    train_set = windowed_dataset(data_train['Close'], WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)

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

    # Load previous executions
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # model.load_weights(filepath=tf.train.latest_checkpoint(checkpoint_dir))

    # Fit the model
    history = model.fit(
        train_set,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1,
                save_weights_only=True,
                period=25),
            tf.keras.callbacks.EarlyStopping(
                monitor='mae',
                min_delta=1,
                patience=5
            )
        ]
    )

    return model


def main():
    # Read dataset from Open Power System data for Germany. Uncomment the appropriate option.

    # File located into local repository
    data_file = os.path.join(os.getcwd(), 'datasets', 'msft_stock_prices.csv')

    # Read daily stock prices dataset from Yahoo and save the result to local file
    df = web.DataReader('MSFT', data_source='yahoo', start='2012-01')
    df.to_csv(data_file, index=True, header=True)

    # Datetime index is created automatically when reading
    # df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Review dataset
    dataset_review(df)

    # Time series visualization
    time_series_visualization(df)

    # Built model for stock price prediction
    model = build_model_stock_price_prediction(df)


if __name__ == '__main__':
    main()
