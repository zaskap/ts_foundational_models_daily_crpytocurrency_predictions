import pandas as pd
import numpy as np
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.base import ForecastingHorizon
import warnings
warnings.filterwarnings('ignore')

class ProphetWrapper:
    def __init__(self, model_params = {}):
        self.model_params = model_params
        self.model = Prophet(**model_params)
        self.is_fitted = False
        print(f"Model initialized: Prophet\n{self.model.get_params()}")

    def fit(self, history_data, frequency=None):
        """Fits the Prophet model.

        Parameters
        ----------
        history_data (pd.Series or pd.DataFrame): Historical time series data with a pandas.DatetimeIndex.
        frequency (str, optional): Frequency of the time series. Defaults to None.
        """
        if not isinstance(history_data.index, pd.DatetimeIndex):
            raise ValueError("Prophet requires the input data to have a pandas.DatetimeIndex.")
        self.model.fit(history_data)
        self.is_fitted = True
        print("Prophet model fitted.")

    def forecast(self, forecasting_horizon=1, future_index=None, X_future=None):
        """Forecasts future values using the fitted Prophet model.

        Parameters
        ----------
        forecasting_horizon (int, optional): Number of steps to forecast. Defaults to 1.
        future_index (pd.Index, optional): Index for the forecast horizon. If None, a relative horizon is used. Defaults to None.
        X_future (pd.DataFrame, optional): Additional regressors for the forecast horizon.

        Returns
        -------
        pd.Series or pd.DataFrame: Point forecasts.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call 'fit' first.")
        if future_index is not None:
            fh = ForecastingHorizon(future_index, is_relative=False)
        else:
            fh = ForecastingHorizon(np.arange(1, forecasting_horizon + 1), is_relative=True)
        point_forecast = self.model.predict(fh=fh)
        return point_forecast.values, None
    
# # Uncomment the following lines to test the code with sample data
# # Sample time series data
# index = pd.date_range("2023-01-01", periods=200, freq="D")
# history_data_single = pd.Series(np.random.randn(200).cumsum(), index=index)

# # Model Initialization
# prophet_params = {
#         'seasonality_mode': "additive",
#         'n_changepoints': int(len(history_data_single) / 12),
#         'yearly_seasonality': 'auto',
#         'weekly_seasonality': 'auto',
#         'daily_seasonality': 'auto',
#         "add_country_holidays":None
#     }

# prophet_forecaster = ProphetWrapper(model_params=prophet_params)

# # Model Fitting
# prophet_forecaster.fit(history_data_single, frequency="D")

# # Model Forecasting
# forecasting_horizon = 7
# point_forecast_prophet, _ = prophet_forecaster.forecast(forecasting_horizon=forecasting_horizon)
# print("Prophet Point Forecast:", point_forecast_prophet)