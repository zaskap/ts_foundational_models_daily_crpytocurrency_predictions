from sktime.forecasting.chronos import ChronosForecaster
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd
import numpy as np

class ChronosWrapper:
    def __init__(self, model_params = {}, forecasting_horizon=1):
        self.forecasting_horizon = forecasting_horizon
        self.model_params = model_params
        self.model = ChronosForecaster(**self.model_params)
        print(f"Model used: ChronosForecaster\nforecasting_horizon={self.forecasting_horizon}\n{self.model.get_params()}")

    def forecast(self, history_data, frequency=None, future_index=None, X_future=None):
        """Forecasts future values using the Chronos model.

        Parameters
        ----------
        history_data (pd.Series or pd.DataFrame): Historical time series data.
        frequency (str, optional): Frequency of the time series. Defaults to None.
        future_index (pd.Index, optional): Index for the forecast horizon. If None, a relative horizon is used. Defaults to None.
        X_future (pd.DataFrame, optional): Exogenous variables for the forecast horizon. Defaults to None.

        Returns
        -------
        pd.Series or pd.DataFrame: Point forecasts.
        """
        self.model.fit(history_data, X=None)

        if future_index is not None:
            fh = ForecastingHorizon(future_index, is_relative=False)
        else:
            fh = np.arange(1, self.forecasting_horizon + 1)

        point_forecast = self.model.predict(fh=fh, X=X_future)
        return point_forecast.values, None

# # Testing the ChronosWrapper class    
# chronos_forecaster = ChronosWrapper(forecasting_horizon=7, model_params = {"model_path":"amazon/chronos-t5-tiny"})
# # Sample time series data
# index = pd.date_range("2023-01-01", periods=200, freq="D")
# history_data_single = pd.Series(np.random.randn(200).cumsum(), index=index)

# point_forecast_chronos, _ = chronos_forecaster.forecast(history_data_single)
# print("Chronos Point Forecast:", point_forecast_chronos)