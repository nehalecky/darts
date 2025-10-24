"""Test ChronosModel implementation with lazy imports."""
from pathlib import Path


def test_chronos_model_file_exists():
    """Verify chronos.py module file exists."""
    chronos_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "chronos.py"
    )

    assert chronos_path.exists(), \
        f"chronos.py not found at {chronos_path}"


def test_chronos_model_has_capability_identifiers():
    """Verify ChronosModel source code defines capability identifiers."""
    chronos_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "chronos.py"
    )

    with open(chronos_path, "r") as f:
        source = f.read()

    # Check that class defines capability identifiers
    assert '_family_name = "chronos"' in source, \
        "ChronosModel should set _family_name = 'chronos'"
    assert '_subfamily_name = "chronos-2"' in source, \
        "ChronosModel should set _subfamily_name = 'chronos-2'"
    assert '_variant_name = None' in source, \
        "ChronosModel should set _variant_name = None (no variants)"


def test_chronos_model_has_lazy_import_check():
    """Verify ChronosModel has lazy import check with helpful error."""
    chronos_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "chronos.py"
    )

    with open(chronos_path, "r") as f:
        source = f.read()

    # Check for lazy import check function
    assert "_check_chronos_available" in source, \
        "Should have _check_chronos_available() function"

    # Check for helpful error message
    assert "uv pip install" in source or "pip install" in source, \
        "Should include installation instructions"
    assert "darts[chronos]" in source, \
        "Should mention darts[chronos] extra"


def test_chronos_model_has_two_layer_validation():
    """Verify ChronosModel implements two-layer validation architecture."""
    chronos_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "chronos.py"
    )

    with open(chronos_path, "r") as f:
        source = f.read()

    # Check for Layer 1 validation call (capability support from registry)
    assert "_validate_capability_support" in source, \
        "Should call _validate_capability_support() for capability validation"

    # Check for Layer 2 validation call (series capabilities)
    assert "_validate_series_capabilities" in source, \
        "Should call _validate_series_capabilities() for series validation"

    # Verify validation happens in _zero_shot_fit
    assert "def _zero_shot_fit" in source, \
        "Should implement _zero_shot_fit method"


def test_base_class_has_capability_validation():
    """Verify FoundationForecastingModel base class has capability validation method."""
    base_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "base.py"
    )

    with open(base_path, "r") as f:
        source = f.read()

    # Check for capability validation method
    assert "def _validate_capability_support" in source, \
        "Base class should define _validate_capability_support() method"

    # Check for registry import
    assert "from .registry import get_model_spec" in source, \
        "Base class should import get_model_spec from registry"

    # Check for capability checks
    assert "past_covariates" in source and "future_covariates" in source, \
        "Should check for covariate capabilities"
