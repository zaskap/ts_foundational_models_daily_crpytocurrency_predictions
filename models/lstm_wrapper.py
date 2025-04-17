import pandas as pd
import numpy as np
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from sktime.forecasting.base import ForecastingHorizon
import warnings
warnings.filterwarnings('ignore')

class NeuralForecastLSTMWrapper:
    def __init__(self, futr_exog_list=None, max_steps=5, model_params = {}):
        self.futr_exog_list = futr_exog_list
        self.max_steps = max_steps
        self.model_params = model_params
        self.model = NeuralForecastLSTM(
            "YE-DEC", futr_exog_list=self.futr_exog_list, max_steps=self.max_steps, **self.model_params
        )
        self.is_fitted = False
        print(f"Model initialized: NeuralForecastLSTM\n{self.model.get_params()}")

    def fit(self, history_data, frequency=None, X=None, fh=None):
        """Fits the NeuralForecastLSTM model.

        Parameters
        ----------
        history_data (pd.Series or pd.DataFrame): Historical target time series data.
        frequency (str, optional): Frequency of the time series. Defaults to None.
        X (pd.DataFrame, optional): Exogenous variables for training. Defaults to None.
        fh (int, list, np.ndarray, ForecastingHorizon, optional): Forecasting horizon. Defaults to None.
        """
        if isinstance(history_data, pd.DataFrame) and history_data.shape[1] > 1 and X is None:
            y = history_data.iloc[:, 0]
            X_train = history_data.iloc[:, 1:]
            self.model.fit(y, X=X_train, fh=fh)
        else:
            self.model.fit(history_data, X=X, fh=fh)
        self.is_fitted = True
        print("NeuralForecastLSTM model fitted.")

    def forecast(self, forecasting_horizon=1, future_index=None, X_future=None):
        """Forecasts future values using the fitted NeuralForecastLSTM model.

        Parameters
        ----------
        forecasting_horizon (int, optional): Number of steps to forecast. Defaults to 1.
        future_index (pd.Index, optional): Index for the forecast horizon. If None, a relative horizon is used. Defaults to None.
        X_future (pd.DataFrame, optional): Exogenous variables for the forecast horizon. Defaults to None.

        Returns
        -------
        pd.Series or pd.DataFrame: Point forecasts.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call 'fit' first.")
        fh = ForecastingHorizon(np.arange(1, forecasting_horizon + 1), is_relative=True)
        point_forecast = self.model.predict(fh=fh, X=X_future)
        return point_forecast.values, None
    

# # Uncomment the following lines to test the code with sample data
# # Sample time series data
# index = pd.date_range("2023-01-01", periods=200, freq="D")
# history_data_single = pd.Series(np.random.randn(200).cumsum(), index=index)

# # Model Initialization
# lstm_forecaster = NeuralForecastLSTMWrapper(max_steps=20)

# # Model Fitting
# forecasting_horizon = 7
# lstm_forecaster.fit(history_data_single, fh=np.arange(1, forecasting_horizon + 1))

# # Forecasting
# point_forecast_lstm, _ = lstm_forecaster.forecast(forecasting_horizon=forecasting_horizon)
# print("NeuralForecastLSTM Point Forecast:", point_forecast_lstm)