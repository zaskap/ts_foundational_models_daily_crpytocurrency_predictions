import timesfm
import pandas as pd
import numpy as np

class TimesFMForecaster:
    def __init__(self, hparams=None, checkpoint=None, backend="gpu"):
        default_hparams = timesfm.TimesFmHparams(
                context_len=128,
                horizon_len=1,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=512,
                backend = backend,
            )
        self.hparams = hparams if hparams else default_hparams
        default_checkpoint = timesfm.TimesFmCheckpoint(version = "torch",
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch")
        self.checkpoint = checkpoint if checkpoint else default_checkpoint
        self.model = timesfm.TimesFm(hparams=self.hparams, checkpoint=self.checkpoint)
        print(f"Model used: {self.model.hparams}")

    def forecast(self, history_data, frequency=None):
        """Forecasts future values using the TimesFM model.

        Parameters
    	---------
            history_data (list of np.ndarray): List of historical time series data.
            frequency (list of int, optional): List of frequencies corresponding to each series. Defaults to None.

        Returns
    	-------
            tuple: Point forecasts and experimental quantile forecasts (if available).
        """
        point_forecast, quantile_forecast = self.model.forecast(history_data, freq=frequency)
        return point_forecast, quantile_forecast