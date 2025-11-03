"""Tests for FoundationForecastingModel base class."""
import pytest
from unittest.mock import MagicMock, patch

from darts.models.forecasting.foundation.base import FoundationForecastingModel


class MockFoundationModel(FoundationForecastingModel):
    """Concrete implementation for testing."""

    def __init__(self, device=None, lora_config=None):
        super().__init__(device=device, lora_config=lora_config)
        self.load_count = 0

    def _load_pretrained_model(self):
        """Mock implementation that tracks calls."""
        self.load_count += 1
        mock_model = MagicMock()
        mock_model.name = "MockModel"
        return mock_model

    def _zero_shot_fit(self, series, past_covariates=None, future_covariates=None, **kwargs):
        """Mock zero-shot fit."""
        return self

    @property
    def supports_multivariate(self):
        return False

    @property
    def min_train_series_length(self):
        return 1


class TestLazyLoading:
    """Test lazy loading behavior of .model property."""

    def test_model_property_lazy_loads_on_first_access(self):
        """Test that .model property loads on first access."""
        model = MockFoundationModel()

        # Model should not be loaded yet
        assert model._is_loaded is False
        assert model._model is None

        # Access .model property
        loaded_model = model.model

        # Model should now be loaded
        assert model._is_loaded is True
        assert model._model is not None
        assert loaded_model.name == "MockModel"
        assert model.load_count == 1

    def test_model_property_caches_after_first_load(self):
        """Test that .model property returns cached model on subsequent calls."""
        model = MockFoundationModel()

        # First access
        first_access = model.model
        assert model.load_count == 1

        # Second access should return same object without reloading
        second_access = model.model
        assert model.load_count == 1  # Should NOT increment
        assert first_access is second_access  # Same object

    def test_model_property_only_calls_load_once(self):
        """Test that _load_pretrained_model is called exactly once."""
        model = MockFoundationModel()

        # Access multiple times
        _ = model.model
        _ = model.model
        _ = model.model

        # Should only load once
        assert model.load_count == 1
