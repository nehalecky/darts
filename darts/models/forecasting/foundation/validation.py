"""
Shared validation utilities for foundation models.

This module provides generic validation functions used across all foundation models
to enforce architectural constraints on context length and forecast horizon.
"""
from darts.logging import raise_if_not


def validate_context_length(
    context_length: int,
    hard_max_context: int,
    patch_size: int,
    logger
) -> None:
    """
    Validate user's context_length against hard model limits.

    Parameters
    ----------
    context_length : int
        User-specified minimum context length
    hard_max_context : int
        Model's architectural maximum context length
    patch_size : int
        Model's patch size (context_length must be divisible by this)
    logger
        Logger instance for error messages

    Raises
    ------
    ValueError
        If context_length violates model constraints
    """
    raise_if_not(
        context_length <= hard_max_context,
        f"context_length={context_length} exceeds model maximum "
        f"of {hard_max_context}",
        logger
    )
    raise_if_not(
        context_length % patch_size == 0,
        f"context_length={context_length} must be divisible by "
        f"patch_size={patch_size}",
        logger
    )
    raise_if_not(
        context_length >= patch_size,
        f"context_length={context_length} must be at least "
        f"patch_size={patch_size}",
        logger
    )


def validate_forecast_horizon(
    max_forecast_horizon: int,
    hard_max_horizon: int,
    patch_size: int,
    logger
) -> None:
    """
    Validate user's max_forecast_horizon against hard model limits.

    Parameters
    ----------
    max_forecast_horizon : int
        User-specified maximum forecast horizon
    hard_max_horizon : int
        Model's architectural maximum forecast horizon
    patch_size : int
        Model's patch size (max_forecast_horizon must be divisible by this)
    logger
        Logger instance for error messages

    Raises
    ------
    ValueError
        If max_forecast_horizon violates model constraints
    """
    raise_if_not(
        max_forecast_horizon <= hard_max_horizon,
        f"max_forecast_horizon={max_forecast_horizon} exceeds model maximum "
        f"of {hard_max_horizon}",
        logger
    )
    raise_if_not(
        max_forecast_horizon % patch_size == 0,
        f"max_forecast_horizon={max_forecast_horizon} must be divisible by "
        f"patch_size={patch_size}",
        logger
    )
    raise_if_not(
        max_forecast_horizon >= patch_size,
        f"max_forecast_horizon={max_forecast_horizon} must be at least "
        f"patch_size={patch_size}",
        logger
    )
