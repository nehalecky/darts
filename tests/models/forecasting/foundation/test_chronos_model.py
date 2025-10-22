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
