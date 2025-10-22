"""
Chronos Foundation Model Implementation

Amazon Chronos family of time series foundation models with lazy imports.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not

from .base import FoundationForecastingModel
from .capabilities import get_variant
from .validation import validate_context_length, validate_forecast_horizon

logger = get_logger(__name__)


def _check_chronos_available():
    """
    Check if chronos-forecasting is available and provide helpful error if not.

    Raises
    ------
    ImportError
        If chronos-forecasting package is not installed, with instructions.
    """
    try:
        import chronos  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'chronos-forecasting' package is required for ChronosModel but is not installed.\n"
            "\n"
            "Install it with:\n"
            "  uv pip install 'darts[chronos]'\n"
            "\n"
            "Or with pip:\n"
            "  pip install 'darts[chronos]'\n"
            "\n"
            "This will install chronos-forecasting>=2.0.0 from PyPI."
        ) from e


def _timeseries_to_chronos_df(
    series: Union[TimeSeries, List[TimeSeries]],
    series_id_prefix: str = "series"
) -> pd.DataFrame:
    """
    Convert Darts TimeSeries to Chronos DataFrame format.

    Chronos expects:
    - id column: identifies different time series
    - timestamp column: datetime information
    - target column(s): values to predict (univariate or multivariate)

    Parameters
    ----------
    series : TimeSeries or List[TimeSeries]
        Input time series
    series_id_prefix : str, default="series"
        Prefix for series IDs

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [id, timestamp, target] or [id, timestamp, target_0, target_1, ...]
    """
    # Normalize to list
    if not isinstance(series, list):
        series_list = [series]
    else:
        series_list = series

    dfs = []
    for idx, ts in enumerate(series_list):
        # Get DataFrame from TimeSeries (using narwhals-based to_dataframe)
        # Get with time as column (not index) for easier manipulation
        ts_df = ts.to_dataframe(time_as_index=False)

        # Add series ID as first column
        ts_df.insert(0, "id", f"{series_id_prefix}_{idx}")

        # Rename time column to "timestamp"
        # The time column is the second column (index 1) after inserting id
        time_col_name = ts_df.columns[1]
        ts_df = ts_df.rename(columns={time_col_name: "timestamp"})

        # Rename value columns to "target" (univariate) or "target_0", "target_1", ... (multivariate)
        value_cols = [col for col in ts_df.columns if col not in ["id", "timestamp"]]
        if len(value_cols) == 1:
            ts_df = ts_df.rename(columns={value_cols[0]: "target"})
        else:
            # Multivariate: target_0, target_1, ...
            rename_map = {old: f"target_{i}" for i, old in enumerate(value_cols)}
            ts_df = ts_df.rename(columns=rename_map)

        dfs.append(ts_df)

    # Concatenate all series
    return pd.concat(dfs, ignore_index=True)


def _chronos_df_to_timeseries(
    pred_df: pd.DataFrame,
    original_series: Union[TimeSeries, List[TimeSeries]],
    n: int
) -> Union[TimeSeries, List[TimeSeries]]:
    """
    Convert Chronos prediction DataFrame back to Darts TimeSeries.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Chronos prediction output with columns: [id, timestamp, quantile columns]
    original_series : TimeSeries or List[TimeSeries]
        Original series for metadata (freq, component names)
    n : int
        Forecast horizon

    Returns
    -------
    TimeSeries or List[TimeSeries]
        Predicted time series in Darts format
    """
    # Normalize original series to list
    was_single = not isinstance(original_series, list)
    if was_single:
        series_list = [original_series]
    else:
        series_list = original_series

    results = []
    unique_ids = pred_df["id"].unique()

    for idx, series_id in enumerate(unique_ids):
        # Get predictions for this series
        series_df = pred_df[pred_df["id"] == series_id].copy()

        # Get original series for metadata
        orig_ts = series_list[idx]

        # Set timestamp as index
        series_df = series_df.set_index("timestamp")

        # Remove id column
        series_df = series_df.drop(columns=["id"], errors="ignore")

        # Convert to TimeSeries
        # Check if we have quantile columns (probabilistic forecast)
        quantile_cols = [col for col in series_df.columns if col.replace(".", "").replace("-", "").isdigit()]

        if len(quantile_cols) > 1:
            # Probabilistic: Create stochastic TimeSeries with quantiles as samples
            # Shape: (time_steps, components, samples)
            # Each quantile becomes a sample
            quantile_values = series_df[quantile_cols].values  # (time_steps, n_quantiles)

            # Reshape to (time_steps, 1 component, n_quantiles samples)
            forecast_values = quantile_values[:, np.newaxis, :]

            forecast_ts = TimeSeries.from_times_and_values(
                times=series_df.index,
                values=forecast_values,
                freq=orig_ts.freq,
                columns=orig_ts.components if orig_ts.n_components == 1 else None
            )
        elif "0.5" in series_df.columns:
            # Deterministic: Use median (0.5 quantile) as point forecast
            forecast_values = series_df["0.5"].values
            forecast_ts = TimeSeries.from_times_and_values(
                times=series_df.index,
                values=forecast_values.reshape(-1, 1),
                freq=orig_ts.freq,
                columns=orig_ts.components if orig_ts.n_components == 1 else None
            )
        else:
            # Fallback: Use first value column
            forecast_values = series_df.iloc[:, 0].values
            forecast_ts = TimeSeries.from_times_and_values(
                times=series_df.index,
                values=forecast_values.reshape(-1, 1),
                freq=orig_ts.freq,
                columns=orig_ts.components if orig_ts.n_components == 1 else None
            )

        results.append(forecast_ts)

    # Return in same format as input
    return results[0] if was_single else results


class ChronosModel(FoundationForecastingModel):
    """
    Amazon Chronos 2 foundation model for time series forecasting.

    Chronos is pre-trained on a massive corpus of time series data,
    enabling zero-shot forecasting on new datasets. Chronos 2 supports
    univariate, multivariate, and covariate-informed forecasting with
    probabilistic outputs via in-context learning.

    Parameters
    ----------
    lora_config : dict, optional
        LoRA configuration for parameter-efficient fine-tuning.

    Examples
    --------
    Zero-shot forecasting:

    >>> from darts.models.forecasting.foundation import ChronosModel
    >>> model = ChronosModel()
    >>> # fit() validates inputs and sets model as ready (no training occurs)
    >>> model.fit(train_series)
    >>> forecast = model.predict(n=24)

    With fine-tuning:

    >>> model = ChronosModel(
    ...     lora_config={"r": 8, "lora_alpha": 16}
    ... )
    >>> model.fit(series=training_data, epochs=10)
    >>> forecast = model.predict(n=24)

    Notes
    -----
    - Requires chronos-forecasting>=2.0.0 package
    - Install with: uv pip install 'darts[chronos]'
    - Supports univariate, multivariate, and covariate-informed forecasting
    - Supports probabilistic forecasting via quantile predictions
    - Uses in-context learning for zero-shot adaptation

    References
    ----------
    .. [1] Ansari et al., "Chronos: Learning the Language of Time Series",
           arXiv:2403.07815, 2024. https://arxiv.org/abs/2403.07815
    """

    # Capability identifiers
    _family_name = "chronos"
    _subfamily_name = "chronos-2"
    _variant_name = None  # No variants for Chronos 2

    def __init__(
        self,
        model_id: str = "s3://autogluon/chronos-2",
        device: str = "auto",
        context_length: Optional[int] = None,
        max_forecast_horizon: Optional[int] = None,
        lora_config: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize Chronos model.

        Parameters
        ----------
        model_id : str, default="s3://autogluon/chronos-2"
            Model identifier for Chronos 2. Can be:
            - S3 path: "s3://autogluon/chronos-2" (120M params, default)
            - HuggingFace: Not currently available for Chronos 2
            - Local path: path to downloaded model
        device : str, default="auto"
            Device to use ("auto", "cuda", "mps", "cpu")
        context_length : int, optional
            MINIMUM context length enforced by this model instance for validation.
            If None, uses the model's default (2048).
            Must be divisible by patch_size (16) and ≤ hard max (8192).
            Example: For 144-point series with start=0.75 backtesting, use 96
            (0.75 × 144 ≈ 108, rounded down to nearest multiple of 16).
        max_forecast_horizon : int, optional
            MAXIMUM forecast horizon enforced by this model instance.
            If None, uses the model's hard limit (1024).
            Must be divisible by patch_size (16) and ≤ hard max (1024).
            Useful for constraining forecasts to your use case requirements.
        lora_config : dict, optional
            LoRA configuration for fine-tuning.
        **kwargs
            Additional arguments passed to FoundationForecastingModel.
        """
        # Check chronos-forecasting is available
        _check_chronos_available()

        super().__init__(lora_config=lora_config, **kwargs)

        # Load hard architectural limits from capabilities registry
        caps = get_variant("chronos", "chronos-2")
        self._hard_max_context = caps["max_context_length"]
        self._hard_max_horizon = caps["max_forecast_horizon"]
        self._patch_size = caps["patch_size"]
        self._default_context_length = caps["default_context_length"]

        # Validate and set user's minimum context_length preference
        if context_length is None:
            self.context_length = self._default_context_length
        else:
            validate_context_length(
                context_length, self._hard_max_context, self._patch_size, logger
            )
            self.context_length = context_length

        # Validate and set user's maximum forecast_horizon preference
        if max_forecast_horizon is None:
            self.max_forecast_horizon = self._hard_max_horizon
        else:
            validate_forecast_horizon(
                max_forecast_horizon, self._hard_max_horizon, self._patch_size, logger
            )
            self.max_forecast_horizon = max_forecast_horizon

        self.model_id = model_id
        self.device = device
        self._pipeline = None  # Lazy loading

    @property
    def pipeline(self):
        """Lazy-load Chronos2Pipeline on first use."""
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            logger.info(f"Loading Chronos2Pipeline from '{self.model_id}'...")
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map=self.device if self.device != "auto" else None
            )
            logger.info("Chronos2Pipeline loaded successfully")

        return self._pipeline

    def _zero_shot_fit(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        **kwargs
    ) -> "ChronosModel":
        """
        Validate inputs for zero-shot inference.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Validation series.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Not supported by Chronos.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Not supported by Chronos.
        **kwargs
            Ignored.

        Returns
        -------
        self
            Validated model.
        """
        # Validate series capabilities
        self._validate_series_capabilities(series)

        logger.info("ChronosModel ready for zero-shot forecasting")

        return self

    @property
    def min_train_samples(self) -> int:
        """Minimum number of samples required for training."""
        return 1  # Chronos 2 can work with minimal context

    @property
    def extreme_lags(self) -> tuple:
        """
        Returns the extreme lags used by the model.

        For Chronos 2:
        - Can use up to context_length historical points
        - No covariates support
        - No output chunk shift

        Returns
        -------
        tuple
            (min_target_lag, max_target_lag, min_past_cov_lag, max_past_cov_lag,
             min_future_cov_lag, max_future_cov_lag, output_chunk_shift)
        """
        return (
            -self.context_length,  # min_target_lag: lookback window
            0,                      # max_target_lag: no future target values
            None,                   # min_past_cov_lag: no past covariates
            None,                   # max_past_cov_lag
            None,                   # min_future_cov_lag: no future covariates
            None,                   # max_future_cov_lag
            0,                      # output_chunk_shift: no shift
        )

    def _target_window_lengths(self) -> tuple:
        """
        Returns the input and output window lengths for the model.
        For Chronos 2: (context_length, 0) - configurable context and arbitrary forecast horizon
        """
        return self.context_length, 0  # 0 means arbitrary forecast horizon

    def _model_encoder_settings(self) -> tuple:
        """
        Returns encoder settings.
        Chronos 2 doesn't use covariates.
        """
        return 0, 0, False, False

    def _apply_peft(self) -> None:
        """
        Apply PEFT configuration to Chronos model.

        Raises
        ------
        NotImplementedError
            PEFT support not yet implemented for Chronos.
        """
        raise NotImplementedError(
            "ChronosModel does not yet support PEFT fine-tuning. "
            "This feature is planned for future releases."
        )

    def _train_with_peft(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        **kwargs
    ) -> "ChronosModel":
        """
        Train PEFT adapters on provided data.

        Raises
        ------
        NotImplementedError
            PEFT training not yet implemented for Chronos.
        """
        raise NotImplementedError(
            "ChronosModel does not yet support PEFT training. "
            "This feature is planned for future releases."
        )

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        num_samples: int = 1,
        **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Generate forecasts using Chronos 2.

        Parameters
        ----------
        n : int
            Number of time steps to forecast.
        series : TimeSeries or List[TimeSeries], optional
            Input series for context. Required for zero-shot usage.
        num_samples : int, default=1
            Number of probabilistic samples to generate.
            When num_samples=1, returns point forecast (median).
            When num_samples>1, returns probabilistic forecast.
        **kwargs
            Additional prediction parameters.

        Returns
        -------
        TimeSeries or List[TimeSeries]
            Forecasted time series.

        Raises
        ------
        ValueError
            If series is None or incompatible with model capabilities.

        Examples
        --------
        >>> model = ChronosModel()
        >>> forecast = model.predict(n=24, series=train_series)
        >>> # Probabilistic forecast
        >>> prob_forecast = model.predict(n=24, series=train_series, num_samples=100)
        """
        # Validate inputs
        raise_if_not(
            series is not None,
            "series is required for zero-shot forecasting with ChronosModel"
        )

        # Convert Darts TimeSeries to Chronos DataFrame format
        logger.debug("Converting TimeSeries to Chronos DataFrame format...")
        context_df = _timeseries_to_chronos_df(series)

        # Determine quantile levels for probabilistic forecasting
        if num_samples > 1:
            # Generate multiple quantiles for probabilistic forecast
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            # Just get the median for point forecast
            quantile_levels = [0.5]

        # Call Chronos2Pipeline.predict_df()
        logger.debug(f"Calling Chronos2Pipeline.predict_df(prediction_length={n}, quantile_levels={quantile_levels})...")
        pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=n,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target" if context_df.columns.tolist().count("target") == 1 else None
        )

        # Convert predictions back to Darts TimeSeries
        logger.debug("Converting predictions back to TimeSeries format...")
        forecast = _chronos_df_to_timeseries(pred_df, series, n)

        logger.info(f"Generated {n}-step forecast successfully")
        return forecast
