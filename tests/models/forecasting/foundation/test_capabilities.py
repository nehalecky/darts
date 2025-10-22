"""Test foundation model capabilities registry loader."""
from pathlib import Path


def test_capabilities_module_exists():
    """Verify capabilities.py module file exists."""
    capabilities_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "capabilities.py"
    )

    assert capabilities_path.exists(), \
        f"capabilities.py not found at {capabilities_path}"
