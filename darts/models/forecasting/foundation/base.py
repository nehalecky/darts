"""
Foundation Forecasting Model Base Class

This module provides the base class for time series foundation models in Darts.
Foundation models are pre-trained on massive datasets and support zero-shot forecasting,
few-shot learning, and parameter-efficient fine-tuning (PEFT).
"""

import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Union

from darts import TimeSeries
from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from .registry import get_model_spec

logger = get_logger(__name__)


class FoundationForecastingModel(GlobalForecastingModel):
    """
    Base class for foundation models with optional PEFT support.

    Foundation models are pre-trained on massive time series datasets, enabling:

    - **Zero-shot forecasting**: Direct prediction without calling fit()
    - **Few-shot learning**: In-context learning from example series
    - **Fine-tuning**: Parameter-efficient adaptation via PEFT (LoRA, Prefix Tuning)

    The fit() method is **optional** for zero-shot usage but required for fine-tuning.
    When `lora_config` is provided, fit() applies PEFT adapters and trains them.

    **Quantile-Based Forecasting:**

    Foundation models use quantile-based probabilistic forecasting, which differs
    from traditional parametric models:

    - **Quantile forecasting**: Predicts discrete probability levels (e.g., 10th, 50th, 90th percentile)
    - **Sample-based forecasting**: Draws stochastic trajectories from parametric distributions
    - Foundation models return quantiles directly, not samples from distributions

    For foundation models, use the `quantiles` parameter in predict() for explicit control
    over which probability levels to forecast.

    Parameters
    ----------
    lora_config : dict, optional
        LoRA configuration following Hugging Face PEFT pattern.
        When provided, enables parameter-efficient fine-tuning.

        Example configuration:
            {
                "r": 8,                          # LoRA rank
                "lora_alpha": 16,                # Scaling factor
                "target_modules": ["qkv_proj"],  # Layers to adapt
                "lora_dropout": 0.05,            # Dropout probability
            }

        See https://huggingface.co/docs/peft for details.

    Examples
    --------
    Zero-shot forecasting (no training):

    >>> from darts.models.foundation import TimesFMModel
    >>> model = TimesFMModel()
    >>> forecast = model.predict(n=12, series=my_series)

    Quantile-based probabilistic forecasting:

    >>> model = ChronosModel()
    >>> # Specify quantiles explicitly (recommended)
    >>> forecast = model.predict(n=24, series=my_series, quantiles=[0.1, 0.5, 0.9])

    Fine-tuning with LoRA:

    >>> model = TimesFMModel(
    ...     lora_config={"r": 8, "lora_alpha": 16}
    ... )
    >>> model.fit(series=training_data, epochs=10)
    >>> forecast = model.predict(n=12)

    Notes
    -----
    Foundation models follow different patterns than traditional Darts models:

    - fit() is optional for zero-shot usage
    - predict() can be called without prior fit() call
    - Fine-tuning uses PEFT to train <1% of parameters efficiently
    - Use `quantiles` parameter instead of `num_samples` for clearer semantics

    References
    ----------
    .. [1] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
           ICLR 2022. https://arxiv.org/abs/2106.09685
    .. [2] Hugging Face PEFT library. https://huggingface.co/docs/peft
    """

    # Capability identifiers - subclasses must override
    _family_name: Optional[str] = None
    _subfamily_name: Optional[str] = None
    _variant_name: Optional[str] = None

    def __init__(self, lora_config: Optional[Dict] = None, **kwargs):
        """
        Initialize foundation forecasting model.

        Parameters
        ----------
        lora_config : dict, optional
            LoRA configuration for parameter-efficient fine-tuning.
        **kwargs
            Additional arguments passed to GlobalForecastingModel.
        """
        super().__init__(**kwargs)
        self.lora_config = lora_config
        self._peft_model = None
        self._is_peft_applied = False

    def fit(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        **kwargs
    ) -> "FoundationForecastingModel":
        """
        Fit the foundation model.

        For zero-shot models (lora_config=None), this validates inputs and loads
        the pre-trained model without training.

        When lora_config is provided, this applies PEFT adapters and trains them
        on the provided series.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Training time series.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates (if supported by model).
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates (if supported by model).
        **kwargs
            Additional training parameters (epochs, learning_rate, etc.)

        Returns
        -------
        self
            Fitted model instance.
        """
        if self.lora_config is not None:
            logger.info("Applying PEFT configuration and fine-tuning model")
            self._apply_peft()
            result = self._train_with_peft(series, past_covariates, future_covariates, **kwargs)
        else:
            logger.info("Zero-shot mode: fit() validates inputs without training")
            result = self._zero_shot_fit(series, past_covariates, future_covariates, **kwargs)

        # Mark model as fitted
        self._fit_called = True
        return result

    @abstractmethod
    def _apply_peft(self) -> None:
        """
        Apply PEFT configuration to the base model.

        This method should:
        1. Load the pre-trained base model if not already loaded
        2. Apply LoRA adapters using Hugging Face PEFT library
        3. Set self._peft_model to the adapter-enhanced model
        4. Set self._is_peft_applied = True

        Raises
        ------
        NotImplementedError
            If PEFT is not yet implemented for this model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet support PEFT fine-tuning"
        )

    @abstractmethod
    def _train_with_peft(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        **kwargs
    ) -> "FoundationForecastingModel":
        """
        Train the PEFT adapters on the provided data.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Training time series.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates.
        **kwargs
            Training parameters (epochs, learning_rate, etc.)

        Returns
        -------
        self
            Model with trained PEFT adapters.

        Raises
        ------
        NotImplementedError
            If PEFT training is not yet implemented for this model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet support PEFT training"
        )

    @abstractmethod
    def _zero_shot_fit(
        self,
        series: Union[TimeSeries, List[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]],
        **kwargs
    ) -> "FoundationForecastingModel":
        """
        Validate inputs for zero-shot inference without training.

        This method should:
        1. Validate input series format
        2. Load pre-trained model if not already loaded
        3. Store series for later use in predict()
        4. Return self without modifying model weights

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Validation/reference time series.
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates.
        **kwargs
            Additional parameters (ignored in zero-shot mode).

        Returns
        -------
        self
            Validated model ready for prediction.
        """
        pass

    @property
    def model_name(self) -> str:
        """
        Get the user-facing display name from registry.

        Returns
        -------
        str
            The display name of the model (e.g., "TimesFM 2.5 200M", "Chronos 2 Base").

        Raises
        ------
        AttributeError
            If capability identifiers are not set on the model class.
        """
        if self._family_name is None or self._subfamily_name is None:
            raise AttributeError(
                f"{self.__class__.__name__} must define _family_name and _subfamily_name"
            )

        spec = get_model_spec(self._get_registry_key())
        return spec["metadata"]["name"]

    @property
    def supports_multivariate(self) -> bool:
        """
        Whether this model supports multivariate time series.

        Returns
        -------
        bool
            True if model can handle multivariate series, False otherwise.

        Raises
        ------
        AttributeError
            If capability identifiers are not set on the model class.
        """
        if self._family_name is None or self._subfamily_name is None:
            raise AttributeError(
                f"{self.__class__.__name__} must define _family_name and _subfamily_name"
            )

        spec = get_model_spec(self._get_registry_key())
        return spec["capabilities"]["multivariate"]

    @property
    def supports_probabilistic(self) -> bool:
        """
        Whether this model supports probabilistic forecasting.

        Returns
        -------
        bool
            True if model can generate probabilistic forecasts, False otherwise.

        Raises
        ------
        AttributeError
            If capability identifiers are not set on the model class.
        """
        if self._family_name is None or self._subfamily_name is None:
            raise AttributeError(
                f"{self.__class__.__name__} must define _family_name and _subfamily_name"
            )

        spec = get_model_spec(self._get_registry_key())
        return spec["capabilities"]["probabilistic"]

    def _validate_series_capabilities(
        self, series: Union[TimeSeries, List[TimeSeries]]
    ) -> None:
        """
        Validate that input series matches model capabilities.

        Parameters
        ----------
        series : TimeSeries or List[TimeSeries]
            Input time series to validate.

        Raises
        ------
        ValueError
            If series capabilities exceed model capabilities (e.g., multivariate
            series provided to univariate-only model).
        """
        # Handle list of series - check first one
        check_series = series[0] if isinstance(series, list) else series

        # Validate multivariate capability
        if not self.supports_multivariate and check_series.width > 1:
            raise ValueError(
                f"{self.__class__.__name__} does not support multivariate time series. "
                f"Input series has {check_series.width} components, but model only supports "
                f"univariate series (1 component). Please provide a univariate series."
            )

    def _validate_capability_support(
        self,
        model_id: str,
        past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    ) -> None:
        """
        Validate that model supports required capabilities based on provided inputs.

        This is the first layer of validation that checks model capabilities from
        the registry. Subclasses should perform second-layer parameter validation
        (context length, quantiles, etc.) separately.

        Parameters
        ----------
        model_id : str
            Model identifier to look up in registry (e.g., "amazon/chronos-t5-base").
        past_covariates : TimeSeries or List[TimeSeries], optional
            Past covariates being used. If provided, model must support past_covariates.
        future_covariates : TimeSeries or List[TimeSeries], optional
            Future covariates being used. If provided, model must support future_covariates.

        Raises
        ------
        ValueError
            If model does not support a required capability.

        Examples
        --------
        >>> # Will raise if TimesFM is used with covariates
        >>> self._validate_capability_support(
        ...     model_id="google/timesfm-2.5-200m",
        ...     past_covariates=past_cov
        ... )
        ValueError: Model does not support past_covariates
        """
        try:
            spec = get_model_spec(model_id)
        except KeyError as e:
            # Re-raise with more helpful context
            raise ValueError(
                f"Model '{model_id}' not found in registry. "
                "Cannot validate capabilities."
            ) from e

        capabilities = spec.get("capabilities", {})

        # Validate past covariates support
        if past_covariates is not None and not capabilities.get("past_covariates", False):
            raise ValueError("Model does not support past_covariates")

        # Validate future covariates support
        if future_covariates is not None and not capabilities.get("future_covariates", False):
            raise ValueError("Model does not support future_covariates")

    def _get_default_quantiles(self, num_samples: int) -> List[float]:
        """
        Get default quantiles based on num_samples parameter.

        Parameters
        ----------
        num_samples : int
            If 1, return median. If >1, return all default quantiles from registry.

        Returns
        -------
        List[float]
            Quantile levels to predict.
        """
        spec = get_model_spec(self._get_registry_key())

        if num_samples == 1:
            # Point forecast: return median only
            return [0.5]
        else:
            # Probabilistic: return default quantiles from registry
            return spec['quantiles']['default']

    def _validate_quantiles_supported(self, quantiles: List[float]) -> None:
        """
        Validate that requested quantiles are supported by the model.

        Parameters
        ----------
        quantiles : List[float]
            Quantiles to validate.

        Raises
        ------
        ValueError
            If any quantile is not in [0, 1] or not supported by model.
        """
        # Basic validation
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles must be in [0, 1]")

        if 0.5 not in quantiles:
            logger.warning(
                "Quantile 0.5 (median) not in requested quantiles. "
                "This may cause issues for point forecast extraction."
            )

        # Registry validation
        spec = get_model_spec(self._get_registry_key())
        supported = spec['quantiles']['supported']

        unsupported = [q for q in quantiles if q not in supported]
        if unsupported:
            raise ValueError(
                f"Model {self._get_registry_key()} does not support quantiles {unsupported}. "
                f"Supported quantiles: {supported}"
            )

    def _supports_true_sampling(self) -> bool:
        """
        Whether this foundation model supports true stochastic sampling.

        Returns
        -------
        bool
            True if model can generate stochastic samples from a distribution.
            False if model only returns discrete quantiles.

        Notes
        -----
        Most current foundation models (Chronos, TimesFM) return quantiles only.
        Future models may support true parametric or non-parametric sampling.
        """
        # Default: foundation models return quantiles, not samples
        # Subclasses can override if they support true sampling
        return False

    @abstractmethod
    def _get_registry_key(self) -> str:
        """
        Get the registry key for this model (e.g., "chronos-2-base").

        Returns
        -------
        str
            Registry key for capability/quantile lookup.
        """
        pass
