# 1. Imports:
import pandas as pd
import numpy as np
from constants import window_size, train_days

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from download_data import get_data_file_name


def get_model_file_name(coin):
    return f"./{coin}_model.h5"

def build_model(coin):
    # Read the dataset:
    df = pd.read_csv(get_data_file_name(coin))

    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df = df[["Date", "Close"]]
    df.index = df["Date"]

    # Sort the dataset on date time and filter “Date” and “Close” columns:
    df = df.sort_index(ascending=True, axis=0)
    df = df.drop("Date", axis=1)
    
    data_ndarray = df.values

    # get range to train data and valid data
    train_data = data_ndarray[0:train_days, :]

    # scale close price to range 0,1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_ndarray)

    x_train_data, y_train_data = [], []

    for i in range(window_size, len(train_data)):
        x_train_data.append(scaled_data[i - window_size : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )

    # Build and train the LSTM model:
    lstm_model = Sequential()
    lstm_model.add(
        LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1))
    )
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

    # 8. Save the LSTM model:
    lstm_model.save(get_model_file_name(coin))
