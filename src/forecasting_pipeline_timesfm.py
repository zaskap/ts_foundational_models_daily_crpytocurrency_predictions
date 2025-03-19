import pandas as pd
import numpy as np
import timesfm

from src.data_loading import load_crypto_data, preprocess_data, split_data
from models.timesfm_wrapper import TimesFMForecaster
from src.evaluation_metrics import *
import os
from config.config import DATA_DIR

OUTPUTS_DIR = os.path.join(DATA_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def forecast_crypto_prices(crypto_symbols, data_dir, train_end_date="2023-12-31"):
    """Forecasts cryptocurrency prices and saves results.

    Parameters
    ---------
        crypto_symbols (list): List of cryptocurrency symbols.
        data_dir (str): Path to the raw data directory.
        train_end_date (str): Date string for the end of the training period.

    Returns
    -------
        dict: Dictionary containing DataFrames with actual and forecasted values.
    """
    all_results = {}

    for symbol in crypto_symbols:
        file_path = os.path.join(data_dir, 'raw', f'{symbol}.csv')
        df = load_crypto_data(file_path)
        if df is None:
            continue

        processed_df = preprocess_data(df)
        if processed_df is None:
            continue

        train_df, test_df = split_data(processed_df, train_end_date)

        if train_df.empty:
            print(f"Warning: No training data available for {symbol}.")
            continue

        history = train_df['Close'].values.astype(np.float32)
        history_list = [history]
        frequency_input = [0]

        #The horizon_len same as the test_data size (366 days - whole year of 2024)
        forecaster = TimesFMForecaster(hparams=timesfm.TimesFmHparams(horizon_len=test_df.shape[0]))
        try:
            point_forecast, _ = forecaster.forecast(history_list, frequency_input)
            forecast_dates = pd.date_range(start=test_df.index.min(), periods=len(point_forecast[0]), freq='D')
            forecast_df = pd.DataFrame({'Forecasted_Close': point_forecast[0]}, index=forecast_dates)

            # Save actual and forecasted values
            output_df = test_df[['Close']].join(forecast_df, how='left')
            output_file_path = os.path.join(OUTPUTS_DIR, f'{symbol}_forecasts.csv')
            output_df.to_csv(output_file_path)
            all_results[symbol] = output_df

            print(f"Forecasts for {symbol} saved to {output_file_path}")

        except Exception as e:
            print(f"Error during forecasting for {symbol}: {e}")
            all_results[symbol] = None

    return all_results


if __name__ == '__main__':
    crypto_symbols = ['BTC-USD', 'BNB-USD', 'ETH-USD', 'SOL-USD']
    data_directory = DATA_DIR
    train_end_date = "2023-12-31"

    if not os.path.exists(os.path.join(data_directory, 'raw')):
        print(f"Error: '{DATA_DIR}\raw' directory not found.")
    else:
        all_data = {}
        train_data = {}
        test_data = {}
        for symbol in crypto_symbols:
            file_path = os.path.join(data_directory, 'raw', f'{symbol}.csv')
            df = load_crypto_data(file_path)
            if df is not None:
                processed_df = preprocess_data(df)
                if processed_df is not None:
                    train_df, test_df = split_data(processed_df, train_end_date)
                    all_data[symbol] = processed_df
                    train_data[symbol] = train_df
                    test_data[symbol] = test_df

        # Generate forecasts and save results
        forecast_results = forecast_crypto_prices(crypto_symbols, data_directory, train_end_date)