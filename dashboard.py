import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from download_data import get_data_file_name
from build_model import get_model_file_name
from constants import train_days, window_size, dashboard_days

coin = "BTC-USD"


def run_dashboard():
    app = dash.Dash()

    scaler = MinMaxScaler(feature_range=(0, 1))

    df = pd.read_csv(get_data_file_name(coin))

    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df["Date"]

    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])

    for i in range(0, len(data)):
        new_data["Date"][i] = data["Date"][i]
        new_data["Close"][i] = data["Close"][i]

    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)

    dataset = new_data.values

    valid = dataset[train_days:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataset)

    model = load_model(get_model_file_name(coin))

    inputs = new_data[len(new_data) - len(valid) - window_size :].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(window_size, inputs.shape[0]):
        X_test.append(inputs[i - window_size : i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    valid = new_data[train_days:]
    valid["Predictions"] = closing_price
    valid = valid[-dashboard_days:]

    app.layout = html.Div(
        [
            html.H1("Cryto Price Analysis Dashboard", style={"textAlign": "center"}),
            dcc.Tabs(
                id="tabs",
                children=[
                    dcc.Tab(
                        label=f"{coin} Stock Data",
                        children=[
                            html.Div(
                                [
                                    html.H2(
                                        "Actual closing price",
                                        style={"textAlign": "center"},
                                    ),
                                    dcc.Graph(
                                        id="Actual Data",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=valid.index,
                                                    y=valid["Close"],
                                                    mode="markers",
                                                )
                                            ],
                                            "layout": go.Layout(
                                                title="scatter plot",
                                                xaxis={"title": "Date"},
                                                yaxis={"title": "Closing Rate"},
                                            ),
                                        },
                                    ),
                                    html.H2(
                                        "LSTM Predicted closing price",
                                        style={"textAlign": "center"},
                                    ),
                                    dcc.Graph(
                                        id="Predicted Data",
                                        figure={
                                            "data": [
                                                go.Scatter(
                                                    x=valid.index,
                                                    y=valid["Predictions"],
                                                    mode="markers",
                                                )
                                            ],
                                            "layout": go.Layout(
                                                title="scatter plot",
                                                xaxis={"title": "Date"},
                                                yaxis={"title": "Closing Rate"},
                                            ),
                                        },
                                    ),
                                ]
                            )
                        ],
                    ),
                ],
            ),
        ]
    )
    app.run_server(debug=True)
