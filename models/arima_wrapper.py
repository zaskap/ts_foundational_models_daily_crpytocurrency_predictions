import pandas as pd
import numpy as np
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
import warnings
warnings.filterwarnings('ignore')

class AutoARIMAWrapper:
    def __init__(self, model_params = {}, suppress_warnings=True):
        self.suppress_warnings = suppress_warnings
        self.model_params = model_params
        self.model = AutoARIMA(
            **self.model_params, suppress_warnings=self.suppress_warnings
        )
        self.is_fitted = False
        print(f"Model initialized: AutoARIMA\n{self.model.get_params()}")

    def fit(self, history_data, frequency=None):
        """Fits the AutoARIMA model.

        Parameters
        ----------
        history_data (pd.Series or pd.DataFrame): Historical time series data.
        frequency (str, optional): Frequency of the time series. Defaults to None.
        """
        self.model.fit(history_data)
        self.is_fitted = True
        print("AutoARIMA model fitted.")

    def forecast(self, forecasting_horizon=1, future_index=None, X_future=None):
        """Forecasts future values using the fitted AutoARIMA model.

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
# model_params = {"sp":1, "d":None, "max_p":5, "max_q":5}
# arima_forecaster = AutoARIMAWrapper(model_params)

# # Model Fitting
# arima_forecaster.fit(history_data_single)

# # Forecasting
# forecasting_horizon = 7
# point_forecast_arima, _ = arima_forecaster.forecast(forecasting_horizon=forecasting_horizon)
# print("ARIMA Point Forecast:", point_forecast_arima)