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

        #Setting horizon_len same as the test_data size
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

def evaluate_forecasts(output_dir, crypto_symbols, forecasted_value_col = "Forecasted_Close",
                       actual_value_col = "Close"):
    """Evaluates the forecasts for train and test data separately.

    Parameters
    ---------
        output_dir (str): location of the output of the forecasting module
        crypto_symbols (list): list of crypto symbols under consideration
        forecasted_value_col (str): Column name for forecasted value in the forecasted output file
        actual_value_col (str): Column name for actual value in the forecasted output file

    Returns
    -------
        dict: Dictionary containing evaluation metrics for each cryptocurrency. We have the provision to
                accommodate the training data evaluation metrics too.
    """
    evaluation_metrics_all = {}
    for symbol in crypto_symbols:
        forecast_result_path = os.path.join(output_dir, f"{symbol}_forecasts.csv")
        forecast_df = pd.read_csv(forecast_result_path)

        if forecast_df is None:
            continue

        evaluation_metrics_all[symbol] = {}

        # Evaluate on Test Data
        evaluation_metrics_all[symbol]['test'] = {}
        test_actual = forecast_df[actual_value_col].dropna().values
        test_predicted = forecast_df[forecasted_value_col].dropna().values

        if len(test_actual) > 0 and len(test_predicted) > 0 and len(test_actual) == len(test_predicted):
            evaluation_metrics_all[symbol]['test']['MAPE'] = calculate_mape(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['ME'] = calculate_me(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['MAE'] = calculate_mae(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['MPE'] = calculate_mpe(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['RMSE'] = calculate_rmse(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['R'] = calculate_r(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['Scalar Product'] = calculate_scalar_product(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['Return Score'] = calculate_return_score(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['Long Return'] = calculate_long_return(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['Short Return'] = calculate_short_return(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['Mean Directional Accuracy'] = calculate_mean_directional_accuracy(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['Mean Directional Accuracy Positive'] = calculate_mean_directional_accuracy_positive(test_actual, test_predicted)
            evaluation_metrics_all[symbol]['test']['Mean Directional Accuracy Negative'] = calculate_mean_directional_accuracy_negative(test_actual, test_predicted)
        else:
            evaluation_metrics_all[symbol]['test']['error'] = "Insufficient or mismatched data for evaluation."

    return evaluation_metrics_all


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

        # Evaluate the forecasts
        evaluation_results = evaluate_forecasts(OUTPUTS_DIR, crypto_symbols,"Forecasted_Close", "Close")

        # Print the evaluation results
        print("\nEvaluation Results:")
        print(evaluation_results)
        for symbol, metrics in evaluation_results.items():
            print(f"\n--- {symbol} ---")
            print("Test Set Metrics:")
            for metric, value in metrics.get('test', {}).items():
                print(f"{metric}: {value}")
            print("Train Set Metrics:") # Note: Train set evaluation might be limited with this forecasting approach
            for metric, value in metrics.get('train', {}).items():
                print(f"{metric}: {value}")