"""
TimesFM Model - Foundation Model Wrapper for Zero-Shot Forecasting

This module provides integration with Google's TimesFM foundation model for time series forecasting.
"""
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not, raise_log

from .base import FoundationForecastingModel

logger = get_logger(__name__)


class TimesFMModel(FoundationForecastingModel):
    """
    TimesFM Foundation Model for Time Series Forecasting
    -----------------------------------------------------

    TODO: Migrate TimesFMModel to inherit from FoundationForecastingModel
          instead of GlobalForecastingModel. This will provide better consistency
          with other foundation models (ChronosModel) and enable shared capabilities
          infrastructure. Currently on legacy infrastructure for backward compatibility.

    This class provides a wrapper around Google's TimesFM foundation model.
    TimesFM is a decoder-only transformer pre-trained on 100B+ time points
    for zero-shot time series forecasting.

    The model supports:
        - Zero-shot forecasting (no training required)
        - Univariate time series
        - Arbitrary forecast horizons
        - GPU acceleration (CUDA/MPS)

    Parameters
    ----------
    model_version : str, default="2.5"
        TimesFM version to use ("1.0", "2.0", or "2.5")
    model_size : str, default="200m"
        Model size ("200m" or "500m" for v2.0+)
    max_context_length : int, default=1024
        Maximum number of historical time points to use as context.
        Must be a multiple of 32 (TimesFM's patch size) and positive.
    zero_shot : bool, default=True
        If True, use pre-trained weights without fine-tuning.
        If False, fine-tuning is applied (not yet implemented).
    device : str, optional
        Device to use ("cpu", "cuda", or "mps").
        If None, automatically detects best available device.
    normalize_inputs : bool, default=True
        Whether to normalize inputs before forecasting
    use_quantile_forecasts : bool, default=False
        Enable probabilistic forecasting via TimesFM's continuous quantile head.
        When True and num_samples > 1, generates uncertainty estimates through
        quantile interpolation. Note: Provides marginal uncertainty at each time
        point but does not preserve temporal correlation across the forecast horizon.

    Examples
    --------
    Zero-shot forecasting on a single series:

    >>> from darts.datasets import AirPassengersDataset
    >>> from darts.models import TimesFMModel
    >>> series = AirPassengersDataset().load()
    >>> model = TimesFMModel(zero_shot=True)
    >>> model.fit(series)
    >>> forecast = model.predict(n=12)

    Batch forecasting on multiple series:

    >>> series_list = [series1, series2, series3]
    >>> model = TimesFMModel()
    >>> model.fit(series_list[0])  # Fit on any series (zero-shot)
    >>> forecasts = model.predict(n=12, series=series_list)

    Notes
    -----
    This model requires the `timesfm` package to be installed:

        git clone https://github.com/google-research/timesfm.git
        cd timesfm
        pip install -e ".[torch]"

    For Apple Silicon users, ensure you have PyTorch with MPS support.

    References
    ----------
    .. [1] Das et al., "A decoder-only foundation model for time-series
           forecasting", ICML 2024.
           https://arxiv.org/abs/2310.10688
    .. [2] Google Research TimesFM: https://github.com/google-research/timesfm
    .. [3] HuggingFace Model: https://huggingface.co/google/timesfm-2.5-200m-pytorch
    """

    # Capability identifiers
    _family_name = "timesfm"
    _subfamily_name = "timesfm-2.5"
    # _variant_name is set dynamically based on model_size

    def __init__(
        self,
        model_version: str = "2.5",
        model_size: str = "200m",
        context_length: Optional[int] = None,
        max_forecast_horizon: Optional[int] = None,
        zero_shot: bool = True,
        device: Optional[str] = None,
        normalize_inputs: bool = True,
        use_quantile_forecasts: bool = False,
        **kwargs
    ):
        # Validate inputs first (before calling super().__init__)
        raise_if_not(
            model_version in ["1.0", "2.0", "2.5"],
            f"model_version must be one of ['1.0', '2.0', '2.5'], got {model_version}",
            logger
        )

        raise_if_not(
            model_size in ["200m", "500m"],
            f"model_size must be one of ['200m', '500m'], got {model_size}",
            logger
        )

        raise_if_not(
            max_context_length > 0,
            f"max_context_length must be positive, got {max_context_length}",
            logger
        )

        raise_if_not(
            max_context_length % 32 == 0,
            f"max_context_length must be divisible by 32, got {max_context_length}",
            logger
        )

        # Set variant name based on model size for capabilities registry
        self._variant_name = model_size

        # Call parent constructor (FoundationForecastingModel)
        super().__init__(**kwargs)

        self.model_version = model_version
        self.model_size = model_size
        self.zero_shot = zero_shot
        self.normalize_inputs = normalize_inputs
        self.use_quantile_forecasts = use_quantile_forecasts

        # Load hard architectural limits from capabilities registry
        caps = get_variant("timesfm", f"timesfm-{model_version}")
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

        # Auto-detect device
        self.device = self._detect_device() if device is None else device

        # Lazy load model (only when needed)
        self._model = None

        logger.info(
            f"Initialized TimesFM {model_version} ({model_size}) "
            f"with context length {self.context_length} on {self.device}"
        )

    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Auto-detected device: {device}")
        return device

    def _load_model(self):
        """Lazy load TimesFM model from HuggingFace"""
        if self._model is not None:
            return

        try:
            import timesfm
        except ImportError:
            raise_log(
                ImportError(
                    "The 'timesfm' package is required for TimesFMModel but is not installed.\n"
                    "\n"
                    "Install it with:\n"
                    "  uv pip install 'darts[timesfm]'\n"
                    "\n"
                    "Or with pip:\n"
                    "  pip install 'darts[timesfm]'\n"
                    "\n"
                    "See INSTALL.md for more details."
                ),
                logger
            )

        logger.info(f"Loading TimesFM {self.model_version} ({self.model_size}) from HuggingFace...")

        try:
            # Set PyTorch precision for better performance
            torch.set_float32_matmul_precision("high")

            # Load TimesFM 2.5 200M model from HuggingFace
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch"
            )

            # Compile model with forecast configuration
            self._model.compile(
                timesfm.ForecastConfig(
                    max_context=self.context_length,
                    max_horizon=256,  # Default, can be overridden in predict
                    normalize_inputs=self.normalize_inputs,
                    use_continuous_quantile_head=self.use_quantile_forecasts,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=self.use_quantile_forecasts,
                )
            )

            logger.info(f"✓ Loaded TimesFM {self.model_version} ({self.model_size})")

        except Exception as e:
            logger.error(f"Failed to load TimesFM model: {e}")
            logger.error("Model loading will be attempted again on first predict() call")
            raise

    @property
    def supports_multivariate(self) -> bool:
        """TimesFM only supports univariate series"""
        return False

    @property
    def supports_probabilistic_prediction(self) -> bool:
        """TimesFM 2.5 supports probabilistic forecasting via quantile head"""
        return self.use_quantile_forecasts

    @property
    def supports_transferable_series_prediction(self) -> bool:
        """TimesFM supports zero-shot forecasting on new series"""
        return True

    @property
    def min_train_series_length(self) -> int:
        """Minimum series length (TimesFM's patch size)"""
        return 32

    @property
    def min_train_samples(self) -> int:
        """Minimum number of samples required for training"""
        return self.min_train_series_length

    @property
    def extreme_lags(self) -> tuple:
        """
        Returns the extreme lags used by the model.

        For TimesFM foundation model:
        - Uses up to max_context_length historical points
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

    def _target_window_lengths(self) -> tuple[int, int]:
        """
        Returns the input and output window lengths for the model.
        For TimesFM: (context_length, arbitrary forecast horizon)
        """
        return self.context_length, 0  # 0 means arbitrary forecast horizon

    def _model_encoder_settings(self) -> tuple[int, int, bool, bool]:
        """
        Returns encoder settings: (lags_past_covariates, lags_future_covariates, uses_static_covariates, uses_future_covariates)
        TimesFM doesn't use covariates in the initial implementation.
        """
        return 0, 0, False, False

    def _zero_shot_fit(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        **kwargs
    ) -> "TimesFMModel":
        """
        Validate inputs for zero-shot inference.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Validation series. Must be univariate.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Not supported by TimesFM.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Not supported by TimesFM.
        **kwargs
            Ignored.

        Returns
        -------
        self
            Validated model.
        """
        # Validate series capabilities using base class method
        self._validate_series_capabilities(series)

        # Validate series length
        series_list = [series] if isinstance(series, TimeSeries) else series
        for s in series_list:
            if len(s) < self.min_train_series_length:
                logger.warning(
                    f"Series has length {len(s)}, which is less than minimum "
                    f"recommended length of {self.min_train_series_length}"
                )

        # Check for unsupported features
        if past_covariates is not None:
            logger.warning("TimesFM does not use past_covariates; they will be ignored")

        if future_covariates is not None:
            logger.warning("Future covariates not yet supported in this version; they will be ignored")

        # Load pre-trained model
        self._load_model()

        logger.info("TimesFMModel ready for zero-shot forecasting")

        return self

    def _apply_peft(self) -> None:
        """
        Apply PEFT configuration to the base model.

        Raises
        ------
        NotImplementedError
            PEFT fine-tuning not yet implemented for TimesFM.
        """
        raise NotImplementedError(
            "TimesFM does not yet support PEFT fine-tuning. "
            "Use zero_shot=True for zero-shot forecasting."
        )

    def _train_with_peft(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        **kwargs
    ) -> "TimesFMModel":
        """
        Train the PEFT adapters on the provided data.

        Raises
        ------
        NotImplementedError
            PEFT training not yet implemented for TimesFM.
        """
        raise NotImplementedError(
            "TimesFM does not yet support PEFT training. "
            "Use zero_shot=True for zero-shot forecasting."
        )

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        num_samples: int = 1,
        **kwargs
    ) -> Union[TimeSeries, List[TimeSeries]]:
        """
        Generate forecasts for n time steps ahead.

        Parameters
        ----------
        n : int
            Forecast horizon (number of time steps to predict)
        series : TimeSeries or List[TimeSeries], optional
            Input series to forecast from. Required for zero-shot mode.
            Must be univariate.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates (not currently supported)
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates (not currently supported)
        num_samples : int, default=1
            Number of samples (only 1 supported currently)

        Returns
        -------
        Union[TimeSeries, List[TimeSeries]]
            Forecasted values. Returns single TimeSeries if input was single,
            otherwise returns list of TimeSeries.
        """
        if series is None:
            raise_log(
                ValueError("series must be provided for prediction in zero-shot mode"),
                logger
            )

        raise_if_not(
            n > 0,
            f"Forecast horizon n must be positive, got {n}",
            logger
        )

        # Lazy load model for zero-shot forecasting
        if self._model is None:
            logger.info("Loading model for zero-shot forecasting...")
            self._load_model()

        # Validate series capabilities
        self._validate_series_capabilities(series)

        # Prepare inputs
        is_single = isinstance(series, TimeSeries)
        series_list = [series] if is_single else series

        # Convert to numpy arrays (flatten to 1D)
        inputs = [s.values().flatten() for s in series_list]

        logger.info(f"Generating forecasts for {len(inputs)} series, horizon={n}")

        # Generate forecasts using TimesFM
        point_forecasts, quantile_forecasts = self._model.forecast(
            horizon=n,
            inputs=inputs,
        )

        # Convert back to TimeSeries
        forecasts = []
        for i, s in enumerate(series_list):
            if self.use_quantile_forecasts and num_samples > 1 and quantile_forecasts is not None:
                # Use quantile forecasts to create probabilistic TimeSeries
                # TimesFM returns: [mean, q10, q20, q30, q40, q50, q60, q70, q80, q90]
                # Shape: (horizon, 10)
                forecast_ts = self._build_probabilistic_forecast_series(
                    quantiles=quantile_forecasts[i],
                    input_series=s,
                    num_samples=num_samples,
                    random_state=kwargs.get('random_state', None),
                )
            else:
                # Deterministic forecast
                forecast_ts = self._build_forecast_series(
                    points=point_forecasts[i],
                    input_series=s,
                )
            forecasts.append(forecast_ts)

        logger.info(f"✓ Generated {len(forecasts)} forecast(s)")

        return forecasts[0] if is_single else forecasts

    def _build_forecast_series(
        self,
        points: np.ndarray,
        input_series: TimeSeries,
    ) -> TimeSeries:
        """
        Build a TimeSeries from forecast points with proper time index.

        Parameters
        ----------
        points : np.ndarray
            Forecast values (1D array)
        input_series : TimeSeries
            Original input series (used to continue time index)

        Returns
        -------
        TimeSeries
            Forecast as a Darts TimeSeries with continued time index
        """
        # Reshape to (n, 1) for univariate
        if points.ndim == 1:
            forecast_values = points.reshape(-1, 1)
        else:
            forecast_values = points

        # Generate time index that continues from input series
        if input_series.has_datetime_index:
            # Continue datetime index using pandas
            start_time = input_series.end_time() + input_series.freq
            time_index = pd.date_range(
                start=start_time,
                periods=len(forecast_values),
                freq=input_series.freq
            )
            forecast_ts = TimeSeries.from_times_and_values(
                times=time_index,
                values=forecast_values,
                columns=input_series.columns
            )
        else:
            # Continue range index
            start_idx = input_series.end_time() + 1
            time_index = pd.RangeIndex(start=start_idx, stop=start_idx + len(forecast_values))
            forecast_ts = TimeSeries.from_times_and_values(
                times=time_index,
                values=forecast_values,
                columns=input_series.columns
            )

        return forecast_ts

    def _build_probabilistic_forecast_series(
        self,
        quantiles: np.ndarray,
        input_series: TimeSeries,
        num_samples: int,
        random_state: Optional[int] = None,
    ) -> TimeSeries:
        """
        Build a probabilistic TimeSeries from quantile forecasts via interpolation.

        TimesFM's quantile head returns: [mean, q10, q20, q30, q40, q50, q60, q70, q80, q90]
        This method converts these quantiles into samples using inverse transform sampling.

        **Method**: Quantile-based uncertainty estimation
        - Samples uniform random quantile levels between 0.1 and 0.9
        - Interpolates between TimesFM's output quantiles
        - Each sample is independent (no temporal correlation across horizon)

        **Limitations**:
        - Marginal distributions only (no joint distribution across time)
        - No trajectory coherence (samples may form unrealistic sequences)
        - Limited to 10th-90th percentile range (no extreme tails)

        For applications requiring temporally correlated scenarios, consider
        models with true simulation capabilities (ARIMA, state space models).

        Parameters
        ----------
        quantiles : np.ndarray
            Quantile forecasts, shape (horizon, 10)
        input_series : TimeSeries
            Original input series (used to continue time index)
        num_samples : int
            Number of samples to generate
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        TimeSeries
            Probabilistic forecast as a Darts TimeSeries with n_samples > 1
        """
        horizon = quantiles.shape[0]

        # Validate and enforce quantile monotonicity
        # TimesFM has fix_quantile_crossing, but verify anyway
        for t in range(horizon):
            if not np.all(np.diff(quantiles[t]) >= 0):
                logger.warning(
                    f"Quantile crossing detected at time step {t}. "
                    f"Sorting to enforce monotonicity."
                )
                quantiles[t] = np.sort(quantiles[t])

        # Generate samples by interpolating between quantiles
        # Using inverse transform sampling from empirical CDF
        quantile_levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Initialize random number generator with seed
        rng = np.random.default_rng(random_state)

        # Sample uniform random quantile levels within observed range [0.1, 0.9]
        # This avoids extrapolation into unobserved tails
        random_quantiles = rng.uniform(0.1, 0.9, size=num_samples)

        # Create samples array: (horizon, 1 component, num_samples)
        samples = np.zeros((horizon, 1, num_samples))

        for t in range(horizon):
            # Interpolate for each time step independently
            # Note: This loses temporal correlation across horizon
            for s in range(num_samples):
                samples[t, 0, s] = np.interp(random_quantiles[s], quantile_levels, quantiles[t])

        # Generate time index that continues from input series
        if input_series.has_datetime_index:
            start_time = input_series.end_time() + input_series.freq
            time_index = pd.date_range(
                start=start_time,
                periods=horizon,
                freq=input_series.freq
            )
        else:
            start_idx = input_series.end_time() + 1
            time_index = pd.RangeIndex(start=start_idx, stop=start_idx + horizon)

        # Build probabilistic TimeSeries
        forecast_ts = TimeSeries.from_times_and_values(
            times=time_index,
            values=samples,
            columns=input_series.columns
        )

        return forecast_ts

    def __str__(self):
        return (
            f"TimesFM(version={self.model_version}, size={self.model_size}, "
            f"context={self.context_length}, device={self.device})"
        )
