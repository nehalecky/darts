"""Test capability validation in FoundationForecastingModel."""
from pathlib import Path
from unittest.mock import Mock
import sys
import importlib.util


def test_validation_method_exists():
    """Verify base class has _validate_series_capabilities method."""
    project_root = Path(__file__).parents[4]

    # Mock dependencies
    sys.modules['darts'] = Mock()
    sys.modules['darts.logging'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'].GlobalForecastingModel = type('GlobalForecastingModel', (), {})

    # Mock capabilities
    capabilities_mock = Mock()
    capabilities_mock.get_variant = Mock(return_value={
        "multivariate": False,
        "probabilistic": True
    })
    sys.modules['darts.models.forecasting.foundation.capabilities'] = capabilities_mock

    # Load base module
    spec = importlib.util.spec_from_file_location(
        "darts.models.forecasting.foundation.base",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "base.py"
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    cls = base.FoundationForecastingModel

    assert hasattr(cls, '_validate_series_capabilities'), \
        "FoundationForecastingModel missing _validate_series_capabilities method"


def test_univariate_model_rejects_multivariate_series():
    """Verify univariate-only model rejects multivariate series."""
    project_root = Path(__file__).parents[4]

    # Mock dependencies
    sys.modules['darts'] = Mock()
    sys.modules['darts.logging'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'].GlobalForecastingModel = type('GlobalForecastingModel', (), {})

    # Mock capabilities - univariate only
    capabilities_mock = Mock()
    capabilities_mock.get_variant = Mock(return_value={
        "multivariate": False,
        "probabilistic": True
    })
    sys.modules['darts.models.forecasting.foundation.capabilities'] = capabilities_mock

    # Load base module
    spec = importlib.util.spec_from_file_location(
        "darts.models.forecasting.foundation.base",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "base.py"
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    # Create test model
    class TestModel(base.FoundationForecastingModel):
        _family_name = "timesfm"
        _subfamily_name = "2.5"
        _variant_name = "200m"

        def _get_registry_key(self):
            return "timesfm-2.5-200m"

        def _zero_shot_fit(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            pass

    model = TestModel()

    # Mock multivariate series (width > 1)
    multivariate_series = Mock()
    multivariate_series.width = 3

    # Should raise ValueError
    try:
        model._validate_series_capabilities(multivariate_series)
        assert False, "Should have raised ValueError for multivariate series"
    except ValueError as e:
        assert "multivariate" in str(e).lower(), \
            "Error message should mention multivariate"


def test_univariate_model_accepts_univariate_series():
    """Verify univariate-only model accepts univariate series."""
    project_root = Path(__file__).parents[4]

    # Mock dependencies
    sys.modules['darts'] = Mock()
    sys.modules['darts.logging'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'].GlobalForecastingModel = type('GlobalForecastingModel', (), {})

    # Mock capabilities - univariate only
    capabilities_mock = Mock()
    capabilities_mock.get_variant = Mock(return_value={
        "multivariate": False,
        "probabilistic": True
    })
    sys.modules['darts.models.forecasting.foundation.capabilities'] = capabilities_mock

    # Load base module
    spec = importlib.util.spec_from_file_location(
        "darts.models.forecasting.foundation.base",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "base.py"
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    # Create test model
    class TestModel(base.FoundationForecastingModel):
        _family_name = "timesfm"
        _subfamily_name = "2.5"
        _variant_name = "200m"

        def _get_registry_key(self):
            return "timesfm-2.5-200m"

        def _zero_shot_fit(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            pass

    model = TestModel()

    # Mock univariate series (width = 1)
    univariate_series = Mock()
    univariate_series.width = 1

    # Should not raise
    try:
        model._validate_series_capabilities(univariate_series)
    except ValueError:
        assert False, "Should not raise ValueError for univariate series"
