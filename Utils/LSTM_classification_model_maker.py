import tensorflow as tf
import numpy as np
import pandas as pd

def create_lstm_classification_model(neurons, dropout_rate, X_train, y_train):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.LSTM(units=neurons, return_sequences=False))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model