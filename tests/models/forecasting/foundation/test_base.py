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

    def _apply_peft(self):
        """Mock PEFT application."""
        pass

    def _train_with_peft(self, series, past_covariates=None, future_covariates=None, **kwargs):
        """Mock PEFT training."""
        return self

    def _get_registry_key(self):
        """Mock registry key."""
        return "mock-model"

    def predict(self, n, series=None, past_covariates=None, future_covariates=None, **kwargs):
        """Mock prediction."""
        return MagicMock()

    def _model_encoder_settings(self):
        """Mock encoder settings."""
        return {}

    def _target_window_lengths(self):
        """Mock target window lengths."""
        return (0, 0)

    @property
    def extreme_lags(self):
        """Mock extreme lags."""
        return (-1, 0, 0, 0, 0, 0)

    @property
    def min_train_samples(self):
        """Mock minimum train samples."""
        return 1

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


class TestDeviceDetection:
    """Test device detection and management."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value='NVIDIA A100')
    def test_device_auto_detection_cuda_when_available(self, mock_name, mock_cuda):
        """Test that CUDA is detected when available."""
        model = MockFoundationModel(device=None)
        assert model.device == "cuda"

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_device_auto_detection_mps_when_cuda_unavailable(self, mock_mps, mock_cuda):
        """Test that MPS is detected when CUDA unavailable."""
        model = MockFoundationModel(device=None)
        assert model.device == "mps"

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_device_auto_detection_cpu_fallback(self, mock_mps, mock_cuda):
        """Test that CPU is used when no GPU available."""
        model = MockFoundationModel(device=None)
        assert model.device == "cpu"

    def test_device_explicit_override(self):
        """Test that explicit device parameter overrides auto-detection."""
        model = MockFoundationModel(device="cpu")
        assert model.device == "cpu"

        model_cuda = MockFoundationModel(device="cuda")
        assert model_cuda.device == "cuda"
