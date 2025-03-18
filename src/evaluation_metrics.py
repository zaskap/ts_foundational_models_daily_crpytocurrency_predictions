import numpy as np
from scipy.stats import pearsonr

def calculate_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: MAPE value. Returns NaN if y_true contains zeros.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def calculate_me(y_true, y_pred):
    """
    Calculates the Mean Error (ME).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: ME value.
    """
    return np.mean(np.array(y_true) - np.array(y_pred))

def calculate_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error (MAE).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: MAE value.
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def calculate_mpe(y_true, y_pred):
    """
    Calculates the Mean Percentage Error (MPE).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: MPE value. Returns NaN if y_true contains zeros.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    return np.mean((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100

def calculate_rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error (RMSE).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: RMSE value.
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def calculate_r(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient (R).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: Pearson correlation coefficient (R). Returns NaN if unable to calculate.
    """
    try:
        corr, _ = pearsonr(y_true, y_pred)
        return corr
    except ValueError:
        return np.nan

def calculate_scalar_product(y_true, y_pred):
    """
    Calculates the scalar product between the true and predicted changes.
    This can indicate the agreement in the direction of change.

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: Scalar product value.
    """
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)
    if len(true_changes) < 1 or len(pred_changes) < 1 or len(true_changes) != len(pred_changes):
        return np.nan
    return np.sum(true_changes * pred_changes)

def calculate_return_score(y_true, y_pred):
    """
    Calculates a simple return score based on the direction of price changes.
    A positive score indicates agreement in the direction of returns.

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: Return score.
    """
    true_returns = np.sign(np.diff(y_true))
    pred_returns = np.sign(np.diff(y_pred))
    if len(true_returns) < 1 or len(pred_returns) < 1 or len(true_returns) != len(pred_returns):
        return np.nan
    return np.sum(true_returns == pred_returns) / len(true_returns)

def calculate_long_return(y_true, y_pred):
    """
    Calculates the accuracy of predicting positive returns (long positions).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: Long return accuracy. Returns NaN if no positive true returns.
    """
    true_returns = np.diff(y_true)
    pred_returns = np.diff(y_pred)
    if len(true_returns) < 1 or len(pred_returns) < 1 or len(true_returns) != len(pred_returns):
        return np.nan
    long_signals_true = true_returns > 0
    if np.sum(long_signals_true) == 0:
        return np.nan
    long_signals_pred = pred_returns > 0
    correct_long_predictions = np.sum(long_signals_true & long_signals_pred)
    return correct_long_predictions / np.sum(long_signals_true)

def calculate_short_return(y_true, y_pred):
    """
    Calculates the accuracy of predicting negative returns (short positions).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: Short return accuracy. Returns NaN if no negative true returns.
    """
    true_returns = np.diff(y_true)
    pred_returns = np.diff(y_pred)
    if len(true_returns) < 1 or len(pred_returns) < 1 or len(true_returns) != len(pred_returns):
        return np.nan
    short_signals_true = true_returns < 0
    if np.sum(short_signals_true) == 0:
        return np.nan
    short_signals_pred = pred_returns < 0
    correct_short_predictions = np.sum(short_signals_true & short_signals_pred)
    return correct_short_predictions / np.sum(short_signals_true)

def calculate_mean_directional_accuracy(y_true, y_pred):
    """
    Calculates the Mean Directional Accuracy (MDA).

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: MDA value. Returns NaN if insufficient data.
    """
    true_directions = np.sign(np.diff(y_true))
    pred_directions = np.sign(np.diff(y_pred))
    if len(true_directions) < 1 or len(pred_directions) < 1 or len(true_directions) != len(pred_directions):
        return np.nan
    correct_directions = np.sum(true_directions == pred_directions)
    return correct_directions / len(true_directions)

def calculate_mean_directional_accuracy_positive(y_true, y_pred):
    """
    Calculates the Mean Directional Accuracy for positive true returns.

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: MDA for positive returns. Returns NaN if no positive true returns.
    """
    true_returns = np.diff(y_true)
    pred_returns = np.diff(y_pred)
    if len(true_returns) < 1 or len(pred_returns) < 1 or len(true_returns) != len(pred_returns):
        return np.nan
    positive_true = true_returns > 0
    if np.sum(positive_true) == 0:
        return np.nan
    predicted_positive = pred_returns > 0
    correct_positive = np.sum(positive_true & predicted_positive)
    return correct_positive / np.sum(positive_true)

def calculate_mean_directional_accuracy_negative(y_true, y_pred):
    """
    Calculates the Mean Directional Accuracy for negative true returns.

    Parameters
    ---------
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns
    -------
        float: MDA for negative returns. Returns NaN if no negative true returns.
    """
    true_returns = np.diff(y_true)
    pred_returns = np.diff(y_pred)
    if len(true_returns) < 1 or len(pred_returns) < 1 or len(true_returns) != len(pred_returns):
        return np.nan
    negative_true = true_returns < 0
    if np.sum(negative_true) == 0:
        return np.nan
    predicted_negative = pred_returns < 0
    correct_negative = np.sum(negative_true & predicted_negative)
    return correct_negative / np.sum(negative_true)

if __name__ == '__main__':
    # Example usage:
    y_true = np.array([10, 12, 15, 13, 16, 18])
    y_pred = np.array([11, 13, 14, 14, 17, 17])

    print(f"MAPE: {calculate_mape(y_true, y_pred):.4f}")
    print(f"ME: {calculate_me(y_true, y_pred):.4f}")
    print(f"MAE: {calculate_mae(y_true, y_pred):.4f}")
    print(f"MPE: {calculate_mpe(y_true, y_pred):.4f}")
    print(f"RMSE: {calculate_rmse(y_true, y_pred):.4f}")
    print(f"R: {calculate_r(y_true, y_pred):.4f}")
    print(f"Scalar Product: {calculate_scalar_product(y_true, y_pred):.4f}")
    print(f"Return Score: {calculate_return_score(y_true, y_pred):.4f}")
    print(f"Long Return: {calculate_long_return(y_true, y_pred):.4f}")
    print(f"Short Return: {calculate_short_return(y_true, y_pred):.4f}")
    print(f"Mean Directional Accuracy: {calculate_mean_directional_accuracy(y_true, y_pred):.4f}")
    print(f"Mean Directional Accuracy Positive: {calculate_mean_directional_accuracy_positive(y_true, y_pred):.4f}")
    print(f"Mean Directional Accuracy Negative: {calculate_mean_directional_accuracy_negative(y_true, y_pred):.4f}")

    y_true_zero = np.array([0, 1, 2])
    y_pred_zero = np.array([0.1, 1.1, 2.1])
    print(f"\nMAPE with zero in y_true: {calculate_mape(y_true_zero, y_pred_zero)}")
    print(f"MPE with zero in y_true: {calculate_mpe(y_true_zero, y_pred_zero)}")

    y_true_same = np.array([1, 1, 1])
    y_pred_diff = np.array([1, 2, 1])
    print(f"\nReturn Score with no change: {calculate_return_score(y_true_same, y_pred_diff)}")
    print(f"Long Return with no positive true return: {calculate_long_return(y_true_same, y_pred_diff)}")
    print(f"Short Return with no negative true return: {calculate_short_return(y_true_same, y_pred_diff)}")
    print(f"MDA Positive with no positive true return: {calculate_mean_directional_accuracy_positive(y_true_same, y_pred_diff)}")
    print(f"MDA Negative with no negative true return: {calculate_mean_directional_accuracy_negative(y_true_same, y_pred_diff)}")