"""
Test that foundation models correctly respect context_length parameter.

This test suite prevents regressions like the bug where Chronos ignored
context_length and used all training data, causing 5.5x MAPE difference
compared to TimesFM on the Energy dataset.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from darts import TimeSeries


class TestContextLengthEnforcement:
    """Test that context_length parameter is properly enforced."""

    def test_chronos_truncates_long_series_to_context_length(self):
        """
        Verify Chronos truncates input series to context_length.

        Regression test for bug where Chronos used all 799 points instead
        of respecting context_length=512, causing poor performance.
        """
        from darts.models.forecasting.foundation.chronos import ChronosModel

        # Create long series (800 points)
        times = pd.date_range('2020-01-01', periods=800, freq='H')
        values = np.random.randn(800, 1)
        long_series = TimeSeries.from_times_and_values(times, values)

        # Initialize with context_length=512
        model = ChronosModel(context_length=512)

        # Mock the pipeline to avoid loading actual model
        mock_pipeline = Mock()
        mock_pipeline.predict_df = Mock(return_value=pd.DataFrame({
            'id': ['series_0'] * 10,
            'timestamp': pd.date_range('2020-02-03 08:00', periods=10, freq='H'),
            '0.5': np.random.randn(10)
        }))

        # Mock _model (the underlying attribute via base class)
        model._model = mock_pipeline
        model._is_loaded = True
        model._fit_called = True

        # Predict with long series
        _ = model.predict(n=10, series=long_series)

        # Check that pipeline received truncated data
        call_args = mock_pipeline.predict_df.call_args
        context_df = call_args[0][0]

        # Count rows for series_0 (should be 512, not 800)
        series_rows = context_df[context_df['id'] == 'series_0']
        assert len(series_rows) == 512, \
            f"Expected 512 rows (context_length), got {len(series_rows)}"

    def test_chronos_uses_last_n_points_not_first_n(self):
        """
        Verify Chronos uses the LAST context_length points, not first.

        This is critical for forecasting - we want the most recent data.
        """
        from darts.models.forecasting.foundation.chronos import ChronosModel

        # Create series with distinctive pattern in last 512 points
        times = pd.date_range('2020-01-01', periods=800, freq='H')
        values = np.ones((800, 1))
        values[-512:] = 999.0  # Mark last 512 points
        series = TimeSeries.from_times_and_values(times, values)

        model = ChronosModel(context_length=512)

        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline.predict_df = Mock(return_value=pd.DataFrame({
            'id': ['series_0'] * 10,
            'timestamp': pd.date_range('2020-02-03 08:00', periods=10, freq='H'),
            '0.5': np.random.randn(10)
        }))

        model._model = mock_pipeline
        model._is_loaded = True
        model._fit_called = True
        _ = model.predict(n=10, series=series)

        # Check that pipeline received the LAST 512 points
        call_args = mock_pipeline.predict_df.call_args
        context_df = call_args[0][0]

        # All target values should be 999.0 (not 1.0)
        target_values = context_df['target'].values
        assert np.all(target_values == 999.0), \
            "Should use last 512 points (value=999), not first 512 (value=1)"

    def test_chronos_truncates_past_covariates_to_match_series(self):
        """
        Verify past_covariates are also truncated to context_length.

        Past covariates must align with the truncated series.
        """
        from darts.models.forecasting.foundation.chronos import ChronosModel

        # Create long series and covariates
        times = pd.date_range('2020-01-01', periods=800, freq='H')
        series = TimeSeries.from_times_and_values(times, np.random.randn(800, 1))
        past_cov = TimeSeries.from_times_and_values(times, np.random.randn(800, 1))

        model = ChronosModel(context_length=512)

        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline.predict_df = Mock(return_value=pd.DataFrame({
            'id': ['series_0'] * 10,
            'timestamp': pd.date_range('2020-02-03 08:00', periods=10, freq='H'),
            '0.5': np.random.randn(10)
        }))

        model._model = mock_pipeline
        model._is_loaded = True
        model._fit_called = True
        _ = model.predict(n=10, series=series, past_covariates=past_cov)

        # Check that past covariates were truncated
        call_args = mock_pipeline.predict_df.call_args
        context_df = call_args[0][0]

        # Should have 512 rows with past_cov columns
        series_rows = context_df[context_df['id'] == 'series_0']
        assert len(series_rows) == 512, \
            f"Past covariates should be truncated to {model.context_length}"
        assert any('past_cov_' in col for col in context_df.columns), \
            "Past covariates should be present in context_df"

    def test_chronos_does_not_truncate_short_series(self):
        """
        Verify Chronos does NOT truncate series shorter than context_length.

        A 300-point series with context_length=512 should pass through unchanged.
        """
        from darts.models.forecasting.foundation.chronos import ChronosModel

        # Create short series (300 < 512)
        times = pd.date_range('2020-01-01', periods=300, freq='H')
        short_series = TimeSeries.from_times_and_values(times, np.random.randn(300, 1))

        model = ChronosModel(context_length=512)

        # Mock the pipeline
        mock_pipeline = Mock()
        mock_pipeline.predict_df = Mock(return_value=pd.DataFrame({
            'id': ['series_0'] * 10,
            'timestamp': pd.date_range('2020-01-13 12:00', periods=10, freq='H'),
            '0.5': np.random.randn(10)
        }))

        model._model = mock_pipeline
        model._is_loaded = True
        model._fit_called = True
        _ = model.predict(n=10, series=short_series)

        # Check that all 300 points were passed
        call_args = mock_pipeline.predict_df.call_args
        context_df = call_args[0][0]

        series_rows = context_df[context_df['id'] == 'series_0']
        assert len(series_rows) == 300, \
            "Short series should not be truncated"


class TestContextLengthConsistency:
    """Test that TimesFM and Chronos have consistent context_length behavior."""

    def test_both_models_use_same_context_length(self):
        """
        Verify TimesFM and Chronos use the same amount of context when
        initialized with the same context_length parameter.

        This ensures fair comparisons between models.
        """
        from darts.models.forecasting.foundation.timesfm import TimesFMModel
        from darts.models.forecasting.foundation.chronos import ChronosModel

        context_length = 512

        # Both models should store the same context_length
        timesfm = TimesFMModel(context_length=context_length)
        chronos = ChronosModel(context_length=context_length)

        assert timesfm.context_length == context_length
        assert chronos.context_length == context_length
        assert timesfm.context_length == chronos.context_length, \
            "Both models should use the same context_length"

    def test_both_models_truncate_long_series_consistently(self):
        """
        Integration test: Verify both models respect context_length setting.

        NOTE: TimesFM and Chronos use different truncation strategies:
        - TimesFM: Passes all data, library truncates internally via max_context
        - Chronos: Darts wrapper truncates before passing to library

        Both strategies are valid as long as context_length is respected.
        This test verifies Chronos truncates (since library doesn't support it).
        """
        from darts.models.forecasting.foundation.timesfm import TimesFMModel
        from darts.models.forecasting.foundation.chronos import ChronosModel

        # Create long series
        times = pd.date_range('2020-01-01', periods=800, freq='H')
        series = TimeSeries.from_times_and_values(times, np.random.randn(800, 1))

        context_length = 512

        # Test Chronos truncation (critical - prevents regression)
        chronos = ChronosModel(context_length=context_length)

        # Mock the Chronos pipeline
        mock_chronos_pipeline = Mock()
        mock_chronos_pipeline.predict_df = Mock(return_value=pd.DataFrame({
            'id': ['series_0'] * 10,
            'timestamp': pd.date_range(series.end_time() + series.freq, periods=10, freq=series.freq),
            '0.5': np.random.randn(10)
        }))

        chronos._model = mock_chronos_pipeline
        chronos._is_loaded = True
        chronos._fit_called = True
        _ = chronos.predict(n=10, series=series)

        # Verify Chronos received truncated input (Darts-level truncation)
        chronos_context_df = mock_chronos_pipeline.predict_df.call_args[0][0]
        chronos_rows = len(chronos_context_df[chronos_context_df['id'] == 'series_0'])
        assert chronos_rows == context_length, \
            f"Chronos should receive {context_length} points, got {chronos_rows}"

        # For TimesFM: context_length is respected by the underlying library
        # (configured at compile time with max_context=self.context_length)
        # No need to test Darts-level truncation since library handles it


@pytest.mark.parametrize("series_length,context_length,expected_length", [
    (100, 512, 100),   # Short series: no truncation
    (512, 512, 512),   # Exact match: no truncation
    (800, 512, 512),   # Long series: truncate to 512
    (1000, 512, 512),  # Very long: truncate to 512
])
def test_context_length_truncation_cases(series_length, context_length, expected_length):
    """
    Parametrized test for various series/context length combinations.

    Ensures truncation logic works correctly across edge cases.
    """
    from darts.models.forecasting.foundation.chronos import ChronosModel

    # Create series of specified length
    times = pd.date_range('2020-01-01', periods=series_length, freq='H')
    series = TimeSeries.from_times_and_values(times, np.random.randn(series_length, 1))

    model = ChronosModel(context_length=context_length)

    # Mock the pipeline
    mock_pipeline = Mock()
    mock_pipeline.predict_df = Mock(return_value=pd.DataFrame({
        'id': ['series_0'] * 10,
        'timestamp': pd.date_range(series.end_time() + series.freq, periods=10, freq=series.freq),
        '0.5': np.random.randn(10)
    }))

    model._model = mock_pipeline
    model._is_loaded = True
    model._fit_called = True
    _ = model.predict(n=10, series=series)

    # Verify correct truncation
    call_args = mock_pipeline.predict_df.call_args
    context_df = call_args[0][0]
    series_rows = context_df[context_df['id'] == 'series_0']

    assert len(series_rows) == expected_length, \
        f"Expected {expected_length} points, got {len(series_rows)}"
