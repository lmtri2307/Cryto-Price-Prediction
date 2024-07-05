from constants import peroid
import yfinance as yf
from pandas import DataFrame


def download_data(coin, peroid=peroid) -> DataFrame:
    data = yf.download(coin, period=peroid, interval="1d")
    return data


def get_data_file_name(coin):
    return f"./{coin}.csv"



