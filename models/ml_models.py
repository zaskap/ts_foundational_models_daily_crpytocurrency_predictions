# models.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
import tensorflow as tf
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from lightgbm import LGBMRegressor
from statsmodels.tsa.vector_ar.var_model import VAR

def create_lstm_model(train_X, lstm_units=20, dense_units=1, dropout=0.0, optimizer='adam', loss='mean_squared_error'):
    """
    Creates and compiles an LSTM model.

    Args:
        train_X (numpy.ndarray): The training input data, used to determine the input shape.
        lstm_units (int, optional): Number of units in the LSTM layer. Defaults to 20.
        dense_units (int, optional): Number of units in the Dense output layer. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        optimizer (str, optional): Optimizer for training. Defaults to 'adam'.
        loss (str, optional): Loss function. Defaults to 'mean_squared_error'.

    Returns:
        keras.models.Sequential: Compiled LSTM model.
    """

    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(train_X.shape[1], train_X.shape[2])))  # Specify input shape
    if dropout > 0.0:
        model.add(Dropout(dropout))
    model.add(Dense(dense_units))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_lstm(model, train_X, train_y, val_X, val_y, epochs=100, batch_size=32,
               early_stopping_patience=10, verbose=0):
    """
    Trains an LSTM model with optional early stopping.

    Args:
        model (keras.models.Sequential): The LSTM model to train.
        train_X (numpy.ndarray): Training input data.
        train_y (numpy.ndarray): Training output data.
        val_X (numpy.ndarray): Validation input data.
        val_y (numpy.ndarray): Validation output data.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Batch size during training. Defaults to 32.
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
        verbose (int, optional): Verbosity mode (0 = silent, 1 = progress bar). Defaults to 0.

    Returns:
        keras.models.Sequential: Trained LSTM model.
        History: Training history.
    """

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=verbose
    )

    history = model.fit(
        train_X, train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_X, val_y),
        verbose=verbose,
        shuffle=False,  # Important for time series
        callbacks=[early_stopping]
    )

    return model, history


def predict_lstm(model, test_X):
    """
    Makes predictions with a trained LSTM model.

    Args:
        model (keras.models.Sequential): Trained LSTM model.
        test_X (numpy.ndarray): Input data for prediction.

    Returns:
        numpy.ndarray: Predicted values.
    """
    return model.predict(test_X)


def create_gru_model(train_X, gru_units=20, dense_units=1, dropout=0.0, optimizer='adam', loss='mean_squared_error'):
    """
    Creates and compiles a GRU model.

    Args:
        train_X (numpy.ndarray): The training input data, used to determine the input shape.
        gru_units (int, optional): Number of units in the GRU layer. Defaults to 20.
        dense_units (int, optional): Number of units in the Dense output layer. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        optimizer (str, optional): Optimizer for training. Defaults to 'adam'.
        loss (str, optional): Loss function. Defaults to 'mean_squared_error'.

    Returns:
        keras.models.Sequential: Compiled GRU model.
    """

    model = Sequential()
    model.add(GRU(units=gru_units, input_shape=(train_X.shape[1], train_X.shape[2])))  # Specify input shape
    if dropout > 0.0:
        model.add(Dropout(dropout))
    model.add(Dense(dense_units))
    model.compile(optimizer=optimizer, loss=loss)
    return model


def train_gru(model, train_X, train_y, val_X, val_y, epochs=100, batch_size=32,
              early_stopping_patience=10, verbose=0):
    """
    Trains a GRU model with optional early stopping.

    Args:
        model (keras.models.Sequential): The GRU model to train.
        train_X (numpy.ndarray): Training input data.
        train_y (numpy.ndarray): Training output data.
        val_X (numpy.ndarray): Validation input data.
        val_y (numpy.ndarray): Validation output data.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Batch size during training. Defaults to 32.
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to 10.
        verbose (int, optional): Verbosity mode (0 = silent, 1 = progress bar). Defaults to 0.

    Returns:
        keras.models.Sequential: Trained GRU model.
        History: Training history.
    """

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=verbose
    )

    history = model.fit(
        train_X, train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_X, val_y),
        verbose=verbose,
        shuffle=False,  # Important for time series
        callbacks=[early_stopping]
    )

    return model, history


def predict_gru(model, test_X):
    """
    Makes predictions with a trained GRU model.

    Args:
        model (keras.models.Sequential): Trained GRU model.
        test_X (numpy.ndarray): Input data for prediction.

    Returns:
        numpy.ndarray: Predicted values.
    """
    return model.predict(test_X)


def create_arima_model(train_data, order=(5, 1, 0)):
    """
    Creates and fits an ARIMA model.

    Args:
        train_data (pd.Series): Training data.
        order (tuple, optional): ARIMA order (p, d, q). Defaults to (5, 1, 0).

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: Fitted ARIMA model.
    """

    model = sm.tsa.arima.ARIMA(train_data, order=order)
    model_fit = model.fit()
    return model_fit


def predict_arima(model_fit, steps):
    """
    Predicts using a fitted ARIMA model.

    Args:
        model_fit (statsmodels.tsa.arima.model.ARIMAResultsWrapper): Fitted ARIMA model.
        steps (int): Number of steps to forecast.

    Returns:
        numpy.ndarray: Forecasted values.
    """

    forecast = model_fit.forecast(steps=steps)
    return forecast.values


def create_mlp_model(input_shape, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, learning_rate='adaptive'):
    """
    Creates and trains an MLPRegressor model.

    Args:
        input_shape (int): Number of features in the input data.
        hidden_layer_sizes (tuple, optional): Sizes of hidden layers.
        activation (str, optional): Activation function.
        solver (str, optional): Optimization solver.
        alpha (float, optional): L2 regularization term parameter.
        learning_rate (str, optional): Learning rate schedule.

    Returns:
        sklearn.neural_network.MLPRegressor: Trained MLPRegressor model.
    """

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=500  # You might need to adjust max_iter
    )
    return model


def train_mlp(model, train_X, train_y):
    """
     Fits the MLPRegressor model to the training data.

     Args:
         model (sklearn.neural_network.MLPRegressor): The MLPRegressor model.
         train_X (numpy.ndarray): Training input data.
         train_y (numpy.ndarray): Training output data.

     Returns:
         sklearn.neural_network.MLPRegressor: Trained MLPRegressor model.
     """
    model.fit(train_X, train_y)
    return model


def predict_mlp(model, test_X):
    """
    Predicts using a trained MLPRegressor model.

    Args:
        model (sklearn.neural_network.MLPRegressor): Trained MLPRegressor model.
        test_X (numpy.ndarray): Input data for prediction.

    Returns:
        numpy.ndarray: Predicted values.
    """
    return model.predict(test_X)


def create_lgbm_model(params=None):
    """
    Creates an LGBMRegressor model with optional parameters.

    Args:
        params (dict, optional): Parameters for LGBMRegressor.

    Returns:
        lightgbm.LGBMRegressor: LGBMRegressor model.
    """
    if params is None:
        params = {}
    model = LGBMRegressor(**params)
    return model


def train_lgbm(model, train_X, train_y, eval_set=None, early_stopping_rounds=10):
    """
    Trains an LGBMRegressor model.

    Args:
        model (lightgbm.LGBMRegressor): LGBMRegressor model.
        train_X (numpy.ndarray): Training input data.
        train_y (numpy.ndarray): Training output data.
        eval_set (list, optional): Evaluation set for early stopping.
        early_stopping_rounds (int, optional): Early stopping rounds.

    Returns:
        lightgbm.LGBMRegressor: Trained LGBMRegressor model.
    """

    model.fit(train_X, train_y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=0)
    return model


def predict_lgbm(model, test_X):
    """
    Predicts using a trained LGBMRegressor model.

    Args:
        model (lightgbm.LGBMRegressor): Trained LGBMRegressor model.
        test_X (numpy.ndarray): Input data for prediction.

    Returns:
        numpy.ndarray: Predicted values.
    """
    return model.predict(test_X)


def create_var_model(train_data, lags=1):
    """
    Creates and fits a VAR model.

    Args:
        train_data (pd.DataFrame): Training data.
        lags (int, optional): Lags for the VAR model.

    Returns:
        statsmodels.tsa.vector_ar.var_model.VARResultsWrapper: Fitted VAR model.
    """
    model = VAR(train_data)
    model_fit = model.fit(maxlags=lags, ic='aic', verbose=0)
    return model_fit


def predict_var(model_fit, steps, y_hist):
    """
    Predicts using a fitted VAR model.

    Args:
        model_fit (statsmodels.tsa.vector_ar.var_model.VARResultsWrapper): Fitted VAR model.
        steps (int): Number of steps to forecast.
        y_hist (numpy.ndarray): Lagged values.

    Returns:
        numpy.ndarray: Forecasted values.
    """
    forecast = model_fit.forecast(y_hist, steps=steps)
    return forecast


def create_adaboost_model(params=None):
    """
    Creates an AdaBoostRegressor model with optional parameters.

    Args:
        params (dict, optional): Parameters for AdaBoostRegressor.

    Returns:
        sklearn.ensemble.AdaBoostRegressor: AdaBoostRegressor model.
    """

    if params is None:
        model = AdaBoostRegressor()
    else:
        model = AdaBoostRegressor(**params)
    return model


def train_adaboost(model, train_X, train_y):
    """
    Trains an AdaBoostRegressor model.

    Args:
        model (sklearn.ensemble.AdaBoostRegressor): AdaBoostRegressor model.
        train_X (numpy.ndarray): Training input data.
        train_y (numpy.ndarray): Training output data.

    Returns:
        sklearn.ensemble.AdaBoostRegressor: Trained AdaBoostRegressor model.
    """
    model.fit(train_X, train_y)
    return model


def predict_adaboost(model, test_X):
    """
    Predicts using a trained AdaBoostRegressor model.

    Args:
        model (sklearn.ensemble.AdaBoostRegressor): Trained AdaBoostRegressor model.
        test_X (numpy.ndarray): Input data for prediction.

    Returns:
        numpy.ndarray: Predicted values.
    """
    return model.predict(test_X)