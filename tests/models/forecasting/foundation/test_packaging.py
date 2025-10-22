"""Test foundation model packaging and dependencies."""
import sys
from pathlib import Path

import yaml


def test_can_read_pyproject_toml():
    """Verify we can read pyproject.toml using tomli/tomllib."""
    # Python 3.11+ has tomllib built-in, <3.11 needs tomli package
    if sys.version_info >= (3, 11):
        import tomllib as tomli
    else:
        import tomli

    # Read pyproject.toml from project root
    pyproject_path = Path(__file__).parents[4] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    # Verify we can read basic structure
    assert isinstance(data, dict)
    assert len(data) > 0


def test_setup_py_contains_tomli_import():
    """Verify setup.py file contains the tomli/tomllib import pattern."""
    setup_path = Path(__file__).parents[4] / "setup.py"

    with open(setup_path, "r") as f:
        setup_content = f.read()

    # Check for the import pattern
    assert "import tomli" in setup_content or "import tomllib" in setup_content, \
        "setup.py missing tomli/tomllib import"


def test_pyproject_has_optional_dependencies():
    """Verify pyproject.toml has [project.optional-dependencies]."""
    if sys.version_info >= (3, 11):
        import tomllib as tomli
    else:
        import tomli

    pyproject_path = Path(__file__).parents[4] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    assert "project" in pyproject, "pyproject.toml missing [project] section"
    assert "optional-dependencies" in pyproject["project"], \
        "pyproject.toml missing [project.optional-dependencies]"


def test_chronos_extra_defined():
    """Verify 'chronos' extra is defined with correct package."""
    if sys.version_info >= (3, 11):
        import tomllib as tomli
    else:
        import tomli

    pyproject_path = Path(__file__).parents[4] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    extras = pyproject["project"]["optional-dependencies"]
    assert "chronos" in extras, "'chronos' extra not defined"
    assert len(extras["chronos"]) > 0, "'chronos' extra is empty"
    assert any("chronos-forecasting" in dep for dep in extras["chronos"]), \
        "'chronos' extra missing chronos-forecasting package"


def test_timesfm_extra_defined():
    """Verify 'timesfm' extra is defined."""
    if sys.version_info >= (3, 11):
        import tomllib as tomli
    else:
        import tomli

    pyproject_path = Path(__file__).parents[4] / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    extras = pyproject["project"]["optional-dependencies"]
    assert "timesfm" in extras, "'timesfm' extra not defined"
    assert len(extras["timesfm"]) > 0, "'timesfm' extra is empty"


def test_capabilities_yaml_exists():
    """Verify capabilities.yaml exists and is valid YAML."""
    capabilities_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "capabilities.yaml"
    )

    assert capabilities_path.exists(), \
        f"capabilities.yaml not found at {capabilities_path}"

    with open(capabilities_path, "r") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "capabilities.yaml must be a dictionary"
    assert len(data) > 0, "capabilities.yaml is empty"


def test_capabilities_has_required_families():
    """Verify capabilities.yaml has chronos and timesfm families."""
    capabilities_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "capabilities.yaml"
    )

    with open(capabilities_path, "r") as f:
        capabilities = yaml.safe_load(f)

    # Check for required families
    assert "chronos" in capabilities, "chronos family not defined"
    assert "timesfm" in capabilities, "timesfm family not defined"


def test_chronos_family_structure():
    """Verify chronos family has correct hierarchical structure."""
    capabilities_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "capabilities.yaml"
    )

    with open(capabilities_path, "r") as f:
        capabilities = yaml.safe_load(f)

    chronos = capabilities["chronos"]

    # Check for subfamilies
    assert "subfamilies" in chronos, "chronos missing 'subfamilies' key"
    assert isinstance(chronos["subfamilies"], dict), \
        "chronos subfamilies must be a dictionary"

    # Check at least one subfamily exists
    assert len(chronos["subfamilies"]) > 0, \
        "chronos must have at least one subfamily"


def test_variant_has_capabilities():
    """Verify variants have required capability flags."""
    capabilities_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "capabilities.yaml"
    )

    with open(capabilities_path, "r") as f:
        capabilities = yaml.safe_load(f)

    # Check chronos variant has capability flags
    chronos = capabilities["chronos"]
    subfamilies = chronos["subfamilies"]

    # Get first subfamily and first variant
    first_subfamily = next(iter(subfamilies.values()))
    assert "variants" in first_subfamily, "subfamily missing 'variants' key"

    variants = first_subfamily["variants"]
    assert len(variants) > 0, "subfamily must have at least one variant"

    first_variant = next(iter(variants.values()))

    # Check for required capability flags
    required_capabilities = [
        "multivariate",
        "probabilistic",
    ]

    for capability in required_capabilities:
        assert capability in first_variant, \
            f"variant missing required capability: {capability}"
        assert isinstance(first_variant[capability], bool), \
            f"capability '{capability}' must be boolean"
