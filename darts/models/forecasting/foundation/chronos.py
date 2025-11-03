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
from .registry import get_model_spec
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
            "This will install chronos-forecasting>=2.0.0 from PyPI.\n"
            "See INSTALL.md for more details."
        ) from e


def _timeseries_to_chronos_df(
    series: Union[TimeSeries, List[TimeSeries]],
    past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    series_id_prefix: str = "series",
    is_future_df: bool = False
) -> pd.DataFrame:
    """
    Convert Darts TimeSeries to Chronos DataFrame format with covariate support.

    Chronos expects:
    - id column: identifies different time series
    - timestamp column: datetime information
    - target column(s): values to predict (univariate or multivariate)
    - covariate columns: past/future external variables

    Parameters
    ----------
    series : TimeSeries or List[TimeSeries]
        Input time series (omit for future_df)
    past_covariates : TimeSeries or List[TimeSeries], optional
        Past covariates to include in context_df
    future_covariates : TimeSeries or List[TimeSeries], optional
        Future covariates to include in both context_df and future_df
    series_id_prefix : str, default="series"
        Prefix for series IDs
    is_future_df : bool, default=False
        If True, create future_df format (no target columns, only covariates)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [id, timestamp, target(s), covariate columns]
    """
    # Normalize to lists
    series_list = [series] if isinstance(series, TimeSeries) else series if series else []
    past_cov_list = [past_covariates] if isinstance(past_covariates, TimeSeries) else past_covariates if past_covariates else []
    future_cov_list = [future_covariates] if isinstance(future_covariates, TimeSeries) else future_covariates if future_covariates else []

    # Determine number of series
    n_series = max(len(series_list), len(past_cov_list), len(future_cov_list))

    # Broadcast single-element lists
    if series_list and len(series_list) == 1 and n_series > 1:
        series_list = series_list * n_series
    if past_cov_list and len(past_cov_list) == 1 and n_series > 1:
        past_cov_list = past_cov_list * n_series
    if future_cov_list and len(future_cov_list) == 1 and n_series > 1:
        future_cov_list = future_cov_list * n_series

    dfs = []
    for idx in range(n_series):
        # Start with ID
        row_data = {"id": f"{series_id_prefix}_{idx}"}

        # Add target series if not future_df
        if not is_future_df and series_list:
            ts = series_list[idx]
            ts_df = ts.to_dataframe(time_as_index=False)

            # Extract timestamp
            time_col_name = ts_df.columns[0]
            ts_df = ts_df.rename(columns={time_col_name: "timestamp"})

            # Rename value columns to target/target_0/target_1...
            value_cols = [col for col in ts_df.columns if col != "timestamp"]
            if len(value_cols) == 1:
                ts_df = ts_df.rename(columns={value_cols[0]: "target"})
            else:
                rename_map = {old: f"target_{i}" for i, old in enumerate(value_cols)}
                ts_df = ts_df.rename(columns=rename_map)

            # Merge into result
            result_df = ts_df.copy()
            result_df.insert(0, "id", row_data["id"])
        else:
            # For future_df, create empty timestamp placeholder
            result_df = pd.DataFrame(row_data, index=[0])

        # Add past covariates (only in context_df, not future_df)
        if not is_future_df and past_cov_list and idx < len(past_cov_list):
            past_cov = past_cov_list[idx]
            past_df = past_cov.to_dataframe(time_as_index=True)
            # Prefix columns to avoid conflicts
            past_df.columns = [f"past_cov_{col}" for col in past_df.columns]
            # Merge on timestamp
            if "timestamp" in result_df.columns:
                result_df = result_df.merge(past_df, left_on="timestamp", right_index=True, how="left")

        # Add future covariates (in both context_df and future_df)
        if future_cov_list and idx < len(future_cov_list):
            future_cov = future_cov_list[idx]
            future_df = future_cov.to_dataframe(time_as_index=True)
            # Keep original column names (shared between context and future)
            if "timestamp" in result_df.columns:
                result_df = result_df.merge(future_df, left_on="timestamp", right_index=True, how="left")

        dfs.append(result_df)

    # Concatenate all series
    return pd.concat(dfs, ignore_index=True)


def _create_future_df(
    series: Union[TimeSeries, List[TimeSeries]],
    future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
    n: int,
    series_id_prefix: str = "series"
) -> Optional[pd.DataFrame]:
    """
    Create future DataFrame for Chronos with future covariate values.

    Generates future timestamps based on series frequency and extracts
    the corresponding future covariate values for the forecast horizon.

    Parameters
    ----------
    series : TimeSeries or List[TimeSeries]
        Target series (used for timestamp generation and frequency)
    future_covariates : TimeSeries, List[TimeSeries], or None
        Future covariates to include in the future DataFrame
    n : int
        Forecast horizon (number of future steps)
    series_id_prefix : str, default="series"
        Prefix for series IDs in the DataFrame

    Returns
    -------
    pd.DataFrame or None
        Future DataFrame with columns [id, timestamp, covariate_columns]
        Returns None if future_covariates is None
    """
    if future_covariates is None:
        return None

    # Normalize inputs to lists
    was_single_series = not isinstance(series, list)
    series_list = [series] if was_single_series else series

    was_single_cov = not isinstance(future_covariates, list)
    cov_list = [future_covariates] if was_single_cov else future_covariates

    # Broadcast single covariate to all series
    if len(cov_list) == 1 and len(series_list) > 1:
        cov_list = cov_list * len(series_list)

    dfs = []
    for idx, (ts, future_cov) in enumerate(zip(series_list, cov_list)):
        series_id = f"{series_id_prefix}_{idx}"

        # Generate future timestamps based on series frequency
        last_timestamp = ts.end_time()
        freq = ts.freq
        future_timestamps = pd.date_range(
            start=last_timestamp + freq,
            periods=n,
            freq=freq
        )

        # Create base DataFrame with id and timestamp
        future_df = pd.DataFrame({
            "id": series_id,
            "timestamp": future_timestamps
        })

        # Extract future covariate values for forecast horizon
        future_cov_df = future_cov.to_dataframe(time_as_index=True)

        # Filter to the forecast horizon timeframe
        future_cov_df = future_cov_df.loc[future_timestamps]

        # Merge covariates with future DataFrame
        future_df = future_df.merge(
            future_cov_df,
            left_on="timestamp",
            right_index=True,
            how="left"
        )

        dfs.append(future_df)

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

        # Pass device to base class for unified device management
        super().__init__(device=device, lora_config=lora_config, **kwargs)

        # Load hard architectural limits from registry
        spec = get_model_spec("chronos-2-base")
        constraints = spec["constraints"]
        self._hard_max_context = constraints["max_context_length"]
        self._hard_max_horizon = constraints["max_forecast_horizon"]
        self._patch_size = constraints["patch_size"]
        self._default_context_length = constraints["default_context_length"]

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
        # Note: self.device now set by base class
        # Note: self._model now managed by base class (not self._pipeline)

    def _load_pretrained_model(self):
        """
        Load Chronos2Pipeline from pretrained source.

        Returns
        -------
        pipeline
            Loaded Chronos2Pipeline ready for inference.
        """
        from chronos import Chronos2Pipeline

        logger.info(f"Loading Chronos2Pipeline from '{self.model_id}'...")

        pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self.device if self.device != "auto" else None
        )

        logger.info("✓ Chronos2Pipeline loaded successfully")
        return pipeline

    def _get_registry_key(self) -> str:
        """Get registry key for Chronos model."""
        return "chronos-2-base"

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
            Past-observed covariates that will be used during prediction.
            Chronos 2 natively supports incorporating past covariates for forecasting.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future-known covariates that will be used during prediction.
            Chronos 2 natively supports incorporating future covariates for forecasting.
        **kwargs
            Ignored.

        Returns
        -------
        self
            Validated model.
        """
        # Layer 1: Validate model capabilities (base class)
        # Use hardcoded registry key since ChronosModel uses custom S3 path
        self._validate_capability_support(
            model_id="chronos-2-base",
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # Layer 2: Validate series capabilities (model-specific, already implemented)
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
        - Supports both past and future covariates
        - Past covariates share the same context window as the target series
        - Future covariates must extend at least to the forecast horizon (validated at predict() time)
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
            -self.context_length,  # min_past_cov_lag: same lookback as target
            0,                      # max_past_cov_lag: up to present
            0,                      # min_future_cov_lag: from present onward
            self.context_length,   # max_future_cov_lag: at least context_length (extendable to forecast horizon at predict() time)
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
        Chronos 2 supports both past and future covariates.
        """
        return -1, -1, True, True  # -1 = unlimited, both covariate types supported

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
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        num_samples: int = 1,
        quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Generate forecasts using Chronos 2 with optional covariate support.

        Chronos 2 natively supports both past and future covariates, enabling
        exogenous variable integration for improved forecasting accuracy.

        Parameters
        ----------
        n : int
            Number of time steps to forecast.
        series : TimeSeries or List[TimeSeries], optional
            Input series for context. Required for zero-shot usage.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates to condition the forecast on.
            These are exogenous variables observed in the historical context.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates known in advance for the forecast horizon.
            These must extend at least n steps beyond the end of series.
        num_samples : int, default=1
            Number of probabilistic samples to generate.
            When num_samples=1, returns point forecast (median).
            When num_samples>1, returns probabilistic forecast.
        quantiles : List[float], optional
            Specific quantile levels to predict. If None, defaults to 9 quantiles
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for probabilistic forecasts
            or [0.5] for point forecasts. Chronos 2 supports up to 21 quantiles.
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
        >>> # Basic forecast
        >>> forecast = model.predict(n=24, series=train_series)
        >>> # Probabilistic forecast
        >>> prob_forecast = model.predict(n=24, series=train_series, num_samples=100)
        >>> # Forecast with covariates
        >>> forecast = model.predict(
        ...     n=24,
        ...     series=train_series,
        ...     past_covariates=past_cov,
        ...     future_covariates=future_cov
        ... )
        >>> # Custom quantile levels
        >>> forecast = model.predict(
        ...     n=24,
        ...     series=train_series,
        ...     quantiles=[0.1, 0.5, 0.9]
        ... )
        """
        # Validate inputs
        raise_if_not(
            series is not None,
            "series is required for zero-shot forecasting with ChronosModel"
        )

        # Truncate series to context_length (use last N points)
        # This ensures consistent behavior with TimesFM and respects user's context_length setting
        if isinstance(series, list):
            truncated_series = [s[-self.context_length:] if len(s) > self.context_length else s for s in series]
        else:
            truncated_series = series[-self.context_length:] if len(series) > self.context_length else series

        # Truncate past_covariates to align with truncated series
        if past_covariates is not None:
            if isinstance(past_covariates, list):
                truncated_past_cov = [pc[-self.context_length:] if len(pc) > self.context_length else pc for pc in past_covariates]
            else:
                truncated_past_cov = past_covariates[-self.context_length:] if len(past_covariates) > self.context_length else past_covariates
        else:
            truncated_past_cov = None

        # Convert Darts TimeSeries to Chronos DataFrame format with covariates
        logger.debug(f"Converting TimeSeries to Chronos DataFrame format (context_length={self.context_length})...")
        context_df = _timeseries_to_chronos_df(truncated_series, truncated_past_cov, future_covariates)

        # Create future DataFrame with future covariates if provided
        future_df = _create_future_df(series, future_covariates, n)

        # Determine quantile levels for probabilistic forecasting
        if quantiles is None:
            if num_samples > 1:
                # Generate multiple quantiles for probabilistic forecast
                quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            else:
                # Just get the median for point forecast
                quantiles = [0.5]

        # Call Chronos2Pipeline.predict_df()
        logger.debug(f"Calling Chronos2Pipeline.predict_df(prediction_length={n}, quantiles={quantiles})...")
        pred_df = self.model.predict_df(
            context_df,
            prediction_length=n,
            quantile_levels=quantiles,  # Note: Chronos library still uses quantile_levels internally
            future_df=future_df,
            id_column="id",
            timestamp_column="timestamp",
            target="target" if context_df.columns.tolist().count("target") == 1 else None
        )

        # Convert predictions back to Darts TimeSeries
        logger.debug("Converting predictions back to TimeSeries format...")
        forecast = _chronos_df_to_timeseries(pred_df, series, n)

        logger.info(f"Generated {n}-step forecast successfully")
        return forecast
