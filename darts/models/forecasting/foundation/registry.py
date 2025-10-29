"""
Foundation Model Registry Interface.

This module provides programmatic access to the foundation model registry,
which serves as the single source of truth for model capabilities, constraints,
and supported features.

The registry interface handles:
- Loading and caching the registry.yaml file
- Querying model specifications by ID
- Filtering models by capabilities
- Calculating context utilization metrics
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Registry file location (same directory as this module)
REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, Any]:
    """
    Load and parse the foundation model registry.

    The registry is cached after first load for performance. The cache
    is automatically invalidated if the Python process restarts.

    Returns
    -------
    Dict[str, Any]
        Parsed registry dictionary containing model specifications.

    Raises
    ------
    FileNotFoundError
        If registry.yaml cannot be found at the expected location.
    yaml.YAMLError
        If the registry file contains invalid YAML syntax.

    Examples
    --------
    >>> registry = load_registry()
    >>> list(registry['models'].keys())
    ['chronos-2-base', 'chronos-2-large', 'timesfm-2.5-200m']
    """
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry file not found at {REGISTRY_PATH}. "
            "The registry.yaml file should be in the same directory as registry.py."
        )

    try:
        with open(REGISTRY_PATH, "r") as f:
            registry = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse registry file at {REGISTRY_PATH}: {e}"
        ) from e

    if not registry or "models" not in registry:
        raise ValueError(
            f"Invalid registry format: 'models' key not found in {REGISTRY_PATH}"
        )

    return registry


def get_model_spec(model_id: str) -> Dict[str, Any]:
    """
    Get the specification for a specific model.

    Accepts both short model keys (e.g., "chronos-2-base") and full
    HuggingFace model IDs (e.g., "amazon/chronos-2-base"). When a full
    ID is provided, it extracts the model key from the last component.

    Parameters
    ----------
    model_id : str
        Model identifier, either short key or full HuggingFace ID.

    Returns
    -------
    Dict[str, Any]
        Model specification dictionary containing metadata, capabilities,
        quantiles, and constraints.

    Raises
    ------
    KeyError
        If the model_id is not found in the registry.

    Examples
    --------
    >>> spec = get_model_spec("amazon/chronos-2-base")
    >>> spec['capabilities']['multivariate']
    True
    >>> spec['constraints']['max_context_length']
    8192
    """
    registry = load_registry()
    models = registry["models"]

    # Extract model key from full HuggingFace ID if needed
    # e.g., "amazon/chronos-2-base" -> "chronos-2-base"
    model_key = model_id.split("/")[-1]

    if model_key not in models:
        available = ", ".join(models.keys())
        raise KeyError(
            f"Model '{model_id}' not found in registry. "
            f"Available models: {available}"
        )

    return models[model_key]


def list_models(**filters) -> List[str]:
    """
    List model IDs that match the specified capability filters.

    Returns full HuggingFace model IDs (e.g., "amazon/chronos-2-base").
    If no filters are provided, returns all models.

    Parameters
    ----------
    **filters
        Capability filters as keyword arguments. Common filters include:
        - univariate : bool
        - multivariate : bool
        - past_covariates : bool
        - future_covariates : bool
        - probabilistic : bool

    Returns
    -------
    List[str]
        List of full model IDs matching the specified filters.

    Examples
    --------
    >>> # Get all multivariate models
    >>> list_models(multivariate=True)
    ['amazon/chronos-2-base', 'amazon/chronos-2-large']

    >>> # Get models with covariate support
    >>> list_models(past_covariates=True, future_covariates=True)
    ['amazon/chronos-2-base', 'amazon/chronos-2-large']

    >>> # Get univariate-only models
    >>> list_models(univariate=True, multivariate=False)
    ['google/timesfm-2.5-200m']
    """
    registry = load_registry()
    models = registry["models"]
    matching_models = []

    for model_key, spec in models.items():
        # Check if model matches all filters
        capabilities = spec.get("capabilities", {})
        matches = True

        for filter_key, filter_value in filters.items():
            if filter_key not in capabilities:
                # If filter key doesn't exist in capabilities, no match
                matches = False
                break
            if capabilities[filter_key] != filter_value:
                matches = False
                break

        if matches:
            # Return full HuggingFace ID from metadata
            model_id = spec["metadata"]["model_id"]
            matching_models.append(model_id)

    return matching_models


@dataclass
class ContextUtilization:
    """
    Context utilization metrics for a foundation model prediction.

    Tracks how much of a model's available context window is being used
    for a specific forecasting scenario. Helps identify underutilization
    or potential context overflow issues.

    Parameters
    ----------
    model_id : str
        Full HuggingFace model ID.
    max_context : int
        Maximum context length supported by the model.
    target_series_points : int
        Number of historical points from the target series.
    num_dimensions : int
        Number of dimensions (1 for univariate, >1 for multivariate/covariates).

    Attributes
    ----------
    total_points : int
        Total data points being fed to the model (target_series_points * num_dimensions).
    utilization_pct : float
        Percentage of context window being used (0-100).
    is_efficient : bool
        Whether utilization is above 50% (efficient context usage).

    Examples
    --------
    >>> util = ContextUtilization(
    ...     model_id="amazon/chronos-2-base",
    ...     max_context=8192,
    ...     target_series_points=500,
    ...     num_dimensions=3
    ... )
    >>> util.total_points
    1500
    >>> util.utilization_pct
    18.31...
    >>> util.is_efficient
    False
    >>> print(util)
    ContextUtilization(amazon/chronos-2-base): 1500/8192 points (18.3%) - Underutilized
    """

    model_id: str
    max_context: int
    target_series_points: int
    num_dimensions: int

    @property
    def total_points(self) -> int:
        """Total data points being fed to the model."""
        return self.target_series_points * self.num_dimensions

    @property
    def utilization_pct(self) -> float:
        """Percentage of context window being used."""
        if self.max_context == 0:
            return 0.0
        return (self.total_points / self.max_context) * 100.0

    @property
    def is_efficient(self) -> bool:
        """Whether utilization is above 50% threshold."""
        return self.utilization_pct > 50.0

    def __str__(self) -> str:
        """Human-readable utilization summary."""
        status = "Efficient" if self.is_efficient else "Underutilized"
        return (
            f"ContextUtilization({self.model_id}): "
            f"{self.total_points}/{self.max_context} points "
            f"({self.utilization_pct:.1f}%) - {status}"
        )
