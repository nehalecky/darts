"""Integration tests for foundation model consistency."""
import pytest
from unittest.mock import MagicMock, patch

from darts.models.forecasting.foundation import TimesFMModel, ChronosModel


class TestCrossModelConsistency:
    """Test that all foundation models use consistent patterns."""

    @patch('chronos.Chronos2Pipeline')
    @patch('timesfm.TimesFM_2p5_200M_torch')
    def test_both_models_use_same_lazy_loading_pattern(self, mock_timesfm, mock_chronos):
        """Test both models use base class lazy loading."""
        # Setup mocks
        mock_timesfm_model = MagicMock()
        mock_timesfm.from_pretrained.return_value = mock_timesfm_model

        mock_chronos_pipeline = MagicMock()
        mock_chronos.from_pretrained.return_value = mock_chronos_pipeline

        # Test TimesFM
        timesfm = TimesFMModel()
        assert timesfm._is_loaded is False
        _ = timesfm.model
        assert timesfm._is_loaded is True

        # Test Chronos
        chronos = ChronosModel()
        assert chronos._is_loaded is False
        _ = chronos.model
        assert chronos._is_loaded is True

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_both_models_use_same_device_detection(self, mock_mps, mock_cuda):
        """Test both models use base class device detection."""
        timesfm = TimesFMModel(device=None)
        chronos = ChronosModel(device=None)

        # Both should detect MPS
        assert timesfm.device == "mps"
        assert chronos.device == "mps"

    def test_both_models_have_consistent_fit_semantics(self):
        """Test both models support optional fit() for zero-shot."""
        timesfm = TimesFMModel()
        chronos = ChronosModel()

        # Both should have _zero_shot_fit method
        assert hasattr(timesfm, '_zero_shot_fit')
        assert hasattr(chronos, '_zero_shot_fit')

        # Both should have fit method from base
        assert hasattr(timesfm, 'fit')
        assert hasattr(chronos, 'fit')

    def test_both_models_implement_load_pretrained_model(self):
        """Test both models implement required abstract method."""
        timesfm = TimesFMModel()
        chronos = ChronosModel()

        # Both should implement _load_pretrained_model
        assert hasattr(timesfm, '_load_pretrained_model')
        assert hasattr(chronos, '_load_pretrained_model')
        assert callable(timesfm._load_pretrained_model)
        assert callable(chronos._load_pretrained_model)
