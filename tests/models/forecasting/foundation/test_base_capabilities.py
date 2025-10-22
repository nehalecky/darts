"""Test capability properties on FoundationForecastingModel."""
from pathlib import Path
from unittest.mock import Mock
import sys
import importlib.util


def test_base_class_has_capability_properties():
    """Verify FoundationForecastingModel has capability property methods."""
    project_root = Path(__file__).parents[4]

    # Mock all dependencies BEFORE loading base module
    sys.modules['darts'] = Mock()
    sys.modules['darts.logging'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'].GlobalForecastingModel = type('GlobalForecastingModel', (), {})

    # Mock capabilities module with a simple get_variant function
    capabilities_mock = Mock()
    capabilities_mock.get_variant = Mock(return_value={
        "multivariate": False,
        "probabilistic": True
    })
    sys.modules['darts.models.forecasting.foundation.capabilities'] = capabilities_mock

    # Now load base.py
    spec = importlib.util.spec_from_file_location(
        "darts.models.forecasting.foundation.base",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "base.py"
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    # Check that class has the capability property methods
    cls = base.FoundationForecastingModel

    assert hasattr(cls, 'supports_multivariate'), \
        "FoundationForecastingModel missing supports_multivariate property"
    assert hasattr(cls, 'supports_probabilistic'), \
        "FoundationForecastingModel missing supports_probabilistic property"


def test_capability_properties_return_bool():
    """Verify capability properties return boolean values."""
    project_root = Path(__file__).parents[4]

    # Mock darts dependencies
    sys.modules['darts'] = Mock()
    sys.modules['darts.logging'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'].GlobalForecastingModel = type('GlobalForecastingModel', (), {})

    # Mock capabilities module
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

    # Create a test model with capability identifiers
    class TestModel(base.FoundationForecastingModel):
        _family_name = "chronos"
        _subfamily_name = "chronos-2"
        _variant_name = "base"

        def _zero_shot_fit(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            pass

    model = TestModel()

    # Check that properties return booleans
    assert isinstance(model.supports_multivariate, bool), \
        "supports_multivariate must return bool"
    assert isinstance(model.supports_probabilistic, bool), \
        "supports_probabilistic must return bool"


def test_capability_properties_match_registry():
    """Verify capability properties match capabilities.yaml values."""
    import yaml

    project_root = Path(__file__).parents[4]

    # Load capabilities directly to get expected values
    capabilities_path = project_root / "darts" / "models" / "forecasting" / "foundation" / "capabilities.yaml"
    with open(capabilities_path, "r") as f:
        caps = yaml.safe_load(f)

    # Chronos 2 has no variants - capabilities are at subfamily level
    expected = caps["chronos"]["subfamilies"]["chronos-2"]

    # Mock darts dependencies
    sys.modules['darts'] = Mock()
    sys.modules['darts.logging'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'].GlobalForecastingModel = type('GlobalForecastingModel', (), {})

    # Load real capabilities module to test actual integration
    import importlib
    capabilities_spec = importlib.util.spec_from_file_location(
        "capabilities",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "capabilities.py"
    )
    capabilities_module = importlib.util.module_from_spec(capabilities_spec)
    capabilities_spec.loader.exec_module(capabilities_module)
    sys.modules['darts.models.forecasting.foundation.capabilities'] = capabilities_module

    # Load base module
    spec = importlib.util.spec_from_file_location(
        "darts.models.forecasting.foundation.base",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "base.py"
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    # Create test model (Chronos 2 has no variants)
    class TestModel(base.FoundationForecastingModel):
        _family_name = "chronos"
        _subfamily_name = "chronos-2"
        _variant_name = None  # No variants for Chronos 2

        def _zero_shot_fit(self, *args, **kwargs):
            return self

        def predict(self, *args, **kwargs):
            pass

    model = TestModel()

    # Verify properties match registry
    assert model.supports_multivariate == expected["multivariate"], \
        f"supports_multivariate should be {expected['multivariate']}"
    assert model.supports_probabilistic == expected["probabilistic"], \
        f"supports_probabilistic should be {expected['probabilistic']}"
