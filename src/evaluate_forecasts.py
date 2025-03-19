from src.evaluation_metrics import *
import os
from config.config import DATA_DIR
import pandas as pd

OUTPUTS_DIR = os.path.join(DATA_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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
    # Evaluate the forecasts
    crypto_symbols = ['BTC-USD', 'BNB-USD', 'ETH-USD', 'SOL-USD']
    evaluation_results = evaluate_forecasts(OUTPUTS_DIR, crypto_symbols, "Forecasted_Close", "Close")

    # Print the evaluation results
    print("\nEvaluation Results:")
    print(evaluation_results)
    for symbol, metrics in evaluation_results.items():
        print(f"\n--- {symbol} ---")
        print("Test Set Metrics:")
        for metric, value in metrics.get('test', {}).items():
            print(f"{metric}: {value}")
        print("Train Set Metrics:")  # Note: Train set evaluation might be limited with this forecasting approach
        for metric, value in metrics.get('train', {}).items():
            print(f"{metric}: {value}")