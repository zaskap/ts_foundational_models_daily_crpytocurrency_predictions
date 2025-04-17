from sktime.forecasting.moirai_forecaster import MOIRAIForecaster
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd
import numpy as np

class MoiraiWrapper:
    def __init__(self, model_params = {}, forecasting_horizon=1):
        self.forecasting_horizon = forecasting_horizon
        self.model_params = model_params
        self.model = MOIRAIForecaster(**self.model_params)
        print(f"Model used: MOIRAIForecaster\nforecasting_horizon={self.forecasting_horizon}\n{self.model.get_params()}")

    def forecast(self, history_data, frequency=None, future_index=None, X_future=None):
        """Forecasts future values using the MOIRAI model.

        Parameters
        ----------
        history_data (pd.DataFrame): Historical time series data (can include target and exogenous variables).
        frequency (str, optional): Frequency of the time series. Defaults to None.
        future_index (pd.Index, optional): Index for the forecast horizon. If None, a relative horizon is used. Defaults to None.
        X_future (pd.DataFrame, optional): Exogenous variables for the forecast horizon. Defaults to None.

        Returns
        -------
        pd.DataFrame: Point forecasts.
        """
        # Assuming history_data contains both target (y) and exogenous (X) if applicable
        if isinstance(history_data, pd.DataFrame):
            y = history_data.iloc[:, 0] # Assuming the first column is the target
            X_train = history_data.iloc[:, 1:] if history_data.shape[1] > 1 else None
        else:
            y = history_data
            X_train = None

        self.model.fit(y, X=X_train)

        if future_index is not None:
            fh = ForecastingHorizon(future_index, is_relative=False)
        else:
            fh = np.arange(1, self.forecasting_horizon + 1)

        point_forecast = self.model.predict(fh=fh, X=X_future)
        return point_forecast.values, None
    

moirai_forecaster = MoiraiWrapper(forecasting_horizon=7, model_params = {"checkpoint_path" : "sktime/moirai-1.0-R-small", "context_length": 512, "deterministic":True})

# Sample time series data
index = pd.date_range("2023-01-01", periods=200, freq="D")
history_data_single = pd.Series(np.random.randn(200).cumsum(), index=index)

point_forecast_moirai, _ = moirai_forecaster.forecast(history_data_single)
print("MOIRAI Point Forecast:", point_forecast_moirai)