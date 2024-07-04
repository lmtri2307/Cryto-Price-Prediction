from constants import coins, peroid
import yfinance as yf
from pandas import DataFrame


def download_data(ticker, peroid=peroid) -> DataFrame:
    data = yf.download(ticker, period=peroid, interval="1d")
    return data

def get_file_name(coin):
    return f"./{coin}.csv"

def download_all_data():
    for coin in coins:
        data = download_data(coin)
        data.to_csv(get_file_name(coin))