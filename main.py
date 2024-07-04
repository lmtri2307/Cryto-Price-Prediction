from dashboard import run_dashboard

# from download_data import get_data_file_name, download_data
from build_model import build_model
from constants import coins

# def download_all_data():
#     for coin in coins:
#         data = download_data(coin)
#         data.to_csv(get_data_file_name(coin))
def build_all_models():
    for coin in coins:
        build_model(coin)

# download_all_data()
# build_all_models()
run_dashboard()
