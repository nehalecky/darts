"""Tests for TimesFM migration to FoundationForecastingModel."""
import pytest
from unittest.mock import MagicMock, patch

from darts.models.forecasting.foundation.timesfm import TimesFMModel
from darts.models.forecasting.foundation.base import FoundationForecastingModel


class TestTimesFMInheritance:
    """Test that TimesFM properly inherits from FoundationForecastingModel."""

    def test_inherits_from_foundation_forecasting_model(self):
        """Test TimesFM inherits from FoundationForecastingModel."""
        assert issubclass(TimesFMModel, FoundationForecastingModel)

    @patch('darts.models.forecasting.foundation.timesfm.TimesFMModel._load_pretrained_model')
    def test_uses_base_class_lazy_loading(self, mock_load):
        """Test that TimesFM uses base class lazy loading mechanism."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        model = TimesFMModel()

        # Should not be loaded yet
        assert model._is_loaded is False

        # Access .model property
        loaded = model.model

        # Should be loaded now
        assert model._is_loaded is True
        mock_load.assert_called_once()

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_uses_auto_detect_device(self, mock_mps, mock_cuda):
        """Test that TimesFM uses base class device detection."""
        model = TimesFMModel(device=None)
        assert model.device == "mps"

    def test_device_explicit_override(self):
        """Test explicit device parameter works."""
        model = TimesFMModel(device="cpu")
        assert model.device == "cpu"
