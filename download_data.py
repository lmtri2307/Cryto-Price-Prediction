from constants import coins, peroid
import yfinance as yf
from pandas import DataFrame


def download_data(ticker, peroid=peroid) -> DataFrame:
    data = yf.download(ticker, period=peroid, interval="1d")
    return data


def get_data_file_name(coin):
    return f"./{coin}.csv"



