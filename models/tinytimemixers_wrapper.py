import pandas as pd
import numpy as np
from sktime.forecasting.ttm import TinyTimeMixerForecaster

class TinyTimeMixersWrapper:
    def __init__(self, model_params = {}, forecasting_horizon=1):
        self.forecasting_horizon = forecasting_horizon
        self.model_params = model_params
        self.model = TinyTimeMixerForecaster(**self.model_params)
        print(f"Model used: TinyTimeMixerForecaster\nforecasting_horizon={self.forecasting_horizon}\n{self.model.get_params()}")

    def forecast(self, history_data, X_exog = None, frequency=None):
        """Forecasts future values using the TinyTimeMixers model.

        Parameters
        ----------
        history_data (pd.Series or pd.DataFrame): Historical time series data.
        X_exog (pd.DataFrame, optional): Exogenous variables for the historical data. Defaults to None.
        frequency (str, optional): Frequency of the time series. Defaults to None.

        Returns
        -------
        pd.Series or pd.DataFrame: Point forecasts.
        """
        self.model.fit(history_data, fh=np.arange(1, self.forecasting_horizon + 1), X=X_exog)
        point_forecast = self.model.predict()
        return point_forecast.values, None

# # Testing the TinyTimeMixers class
# ttm_forecaster = TinyTimeMixersWrapper(forecasting_horizon=7, model_params = {"model_path":"ibm/TTM", "revision":"main"})
# # Sample time series data
# index = pd.date_range("2023-01-01", periods=200, freq="D")
# history_data_single = pd.Series(np.random.randn(200).cumsum(), index=index)
# point_forecast_ttm, _ = ttm_forecaster.forecast(history_data_single)
# print("TinyTimeMixers Point Forecast:", point_forecast_ttm)


# Sample time series data with exogenous variables and future index
# ttm_forecaster = TinyTimeMixersWrapper(forecasting_horizon=7, model_params = {"model_path":"ibm/TTM", "revision":"main"})
# index = pd.date_range("2023-01-01", periods=200, freq="D")
# history_data_multi = pd.DataFrame({
#     "target": np.random.randn(200).cumsum(),
#     "exog1": np.random.randn(200),
#     "exog2": np.random.randn(200)
# }, index=index)

# point_forecast_ttm_future_without_exog, _ = ttm_forecaster.forecast(history_data_multi["target"])
# print("TTM without exogenous variable Point Forecast:", point_forecast_ttm_future_without_exog)

# point_forecast_ttm_future_with_exog, _ = ttm_forecaster.forecast(history_data_multi["target"], X_exog=history_data_multi["exog1"])
# print("TTM with exogenous variable Point Forecast:", point_forecast_ttm_future_with_exog)

# point_forecast_ttm_future_with_multiple_exog, _ = ttm_forecaster.forecast(history_data_multi["target"], X_exog=history_data_multi[["exog1", "exog2"]])
# print("TTM with Multiple exogenous variable Point Forecast:", point_forecast_ttm_future_with_multiple_exog)