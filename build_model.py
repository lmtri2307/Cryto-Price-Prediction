# 1. Imports:
import pandas as pd
import numpy as np
from constants import window_size
# To run plt in jupyter or gg colab envirionment
# %matplotlib inline
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from download_data import get_data_file_name

def get_model_file_name(coin):
    return f"./{coin}_model.h5"

def build_model(coin):
    # 2. Read the dataset:
    df = pd.read_csv(get_data_file_name(coin))

    # 3. Analyze the closing prices from dataframe:
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df["Date"]

    # 4. Sort the dataset on date time and filter “Date” and “Close” columns:
    data = df.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])

    for i in range(0, len(data)):
        new_dataset["Date"][i] = data["Date"][i]
        new_dataset["Close"][i] = data["Close"][i]

    # 5. Normalize the new filtered dataset:
    # get close price column
    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)
    final_dataset = new_dataset.values

    # get range to train data and valid data
    train_data = final_dataset

    # scale close price to range 0,1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(window_size, len(train_data)):
        x_train_data.append(scaled_data[i - window_size : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )

    # 6. Build and train the LSTM model:
    lstm_model = Sequential()
    lstm_model.add(
        LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1))
    )
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    inputs_data = new_dataset[len(new_dataset) - window_size :].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

    # 8. Save the LSTM model:
    lstm_model.save(get_model_file_name(coin))