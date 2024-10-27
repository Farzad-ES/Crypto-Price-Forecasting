import tensorflow as tf
import numpy as np
import pandas as pd

def lstm_reg_model_maker(neurons, drop_rate, X_train, y_train):
    lstm_model=tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(tf.keras.layers.Dropout(rate=drop_rate))
    lstm_model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    lstm_model.add(tf.keras.layers.Dropout(rate=drop_rate))
    lstm_model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    lstm_model.add(tf.keras.layers.Dropout(rate=drop_rate))
    lstm_model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    lstm_model.add(tf.keras.layers.Dropout(rate=drop_rate))
    lstm_model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    lstm_model.add(tf.keras.layers.Dropout(rate=drop_rate))
    lstm_model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=False))
    lstm_model.add(tf.keras.layers.Dropout(rate=drop_rate))
    lstm_model.add(tf.keras.layers.Dense(units=1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    return lstm_model