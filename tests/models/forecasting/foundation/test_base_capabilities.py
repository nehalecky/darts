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

    # Mock registry module with get_model_spec function
    registry_mock = Mock()
    registry_mock.get_model_spec = Mock(return_value={
        "capabilities": {
            "multivariate": False,
            "probabilistic": True
        }
    })
    sys.modules['darts.models.forecasting.foundation.registry'] = registry_mock

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

    # Mock registry module
    registry_mock = Mock()
    registry_mock.get_model_spec = Mock(return_value={
        "capabilities": {
            "multivariate": False,
            "probabilistic": True
        }
    })
    sys.modules['darts.models.forecasting.foundation.registry'] = registry_mock

    # Load base module
    spec = importlib.util.spec_from_file_location(
        "darts.models.forecasting.foundation.base",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "base.py"
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    # Create a test model with direct model_id
    class TestModel(base.FoundationForecastingModel):
        _family_name = "chronos"
        _subfamily_name = "2"
        _variant_name = "base"

        def _get_registry_key(self):
            return "chronos-2-base"

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
    """Verify capability properties match registry.yaml values."""
    import yaml

    project_root = Path(__file__).parents[4]

    # Load registry directly to get expected values
    registry_path = project_root / "darts" / "models" / "forecasting" / "foundation" / "registry.yaml"
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    # Get expected capabilities for Chronos 2 base model
    # Registry uses short keys, not full HuggingFace IDs
    expected = registry["models"]["chronos-2-base"]["capabilities"]

    # Mock darts dependencies
    sys.modules['darts'] = Mock()
    sys.modules['darts.logging'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'] = Mock()
    sys.modules['darts.models.forecasting.forecasting_model'].GlobalForecastingModel = type('GlobalForecastingModel', (), {})

    # Load real registry module to test actual integration
    import importlib
    registry_spec = importlib.util.spec_from_file_location(
        "registry",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "registry.py"
    )
    registry_module = importlib.util.module_from_spec(registry_spec)
    registry_spec.loader.exec_module(registry_module)
    sys.modules['darts.models.forecasting.foundation.registry'] = registry_module

    # Load base module
    spec = importlib.util.spec_from_file_location(
        "darts.models.forecasting.foundation.base",
        project_root / "darts" / "models" / "forecasting" / "foundation" / "base.py"
    )
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    # Create test model using direct model_id
    class TestModel(base.FoundationForecastingModel):
        _family_name = "chronos"
        _subfamily_name = "2"
        _variant_name = "base"

        def _get_registry_key(self):
            return "chronos-2-base"

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
