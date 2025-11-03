"""Tests for Chronos migration to unified base class pattern."""
import pytest
from unittest.mock import MagicMock, patch

from darts.models.forecasting.foundation import ChronosModel


class TestChronosUnifiedPattern:
    """Test that Chronos uses unified base class pattern."""

    @patch('darts.models.forecasting.foundation.chronos.Chronos2Pipeline')
    def test_model_property_returns_chronos_pipeline(self, mock_pipeline_class):
        """Test that .model property returns Chronos2Pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        model = ChronosModel()

        # Access .model property
        loaded = model.model

        # Should return Chronos2Pipeline
        assert loaded is mock_pipeline
        mock_pipeline_class.from_pretrained.assert_called_once()

    @patch('darts.models.forecasting.foundation.chronos.Chronos2Pipeline')
    def test_load_pretrained_model_uses_device_map(self, mock_pipeline_class):
        """Test that _load_pretrained_model passes device_map correctly."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test with explicit device
        model = ChronosModel(device="cuda")
        _ = model.model

        # Should pass device to from_pretrained
        call_kwargs = mock_pipeline_class.from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "cuda"

    @patch('darts.models.forecasting.foundation.chronos.Chronos2Pipeline')
    def test_model_property_caches_pipeline(self, mock_pipeline_class):
        """Test that .model property caches after first load."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        model = ChronosModel()

        # First access
        first = model.model
        # Second access
        second = model.model

        # Should only load once
        assert mock_pipeline_class.from_pretrained.call_count == 1
        assert first is second
