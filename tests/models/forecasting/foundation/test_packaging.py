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


def test_registry_yaml_exists():
    """Verify registry.yaml exists and is valid YAML."""
    registry_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "registry.yaml"
    )

    assert registry_path.exists(), \
        f"registry.yaml not found at {registry_path}"

    with open(registry_path, "r") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "registry.yaml must be a dictionary"
    assert len(data) > 0, "registry.yaml is empty"


def test_registry_has_required_models():
    """Verify registry.yaml has required model entries."""
    registry_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "registry.yaml"
    )

    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    # Check for models section
    assert "models" in registry, "registry missing 'models' key"
    models = registry["models"]

    # Check for required model keys (short form, not full HuggingFace IDs)
    # Registry uses short keys, get_model_spec() handles extracting them from full IDs
    required_model_keys = [
        "chronos-2-base",
        "chronos-2-large",
        "timesfm-2.5-200m",
    ]

    for model_key in required_model_keys:
        assert model_key in models, f"registry missing required model: {model_key}"


def test_registry_model_structure():
    """Verify registry models have correct structure."""
    registry_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "registry.yaml"
    )

    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    models = registry["models"]

    # Check each model has required sections
    for model_id, model_spec in models.items():
        assert "metadata" in model_spec, \
            f"model {model_id} missing 'metadata' section"
        assert "capabilities" in model_spec, \
            f"model {model_id} missing 'capabilities' section"
        assert "constraints" in model_spec, \
            f"model {model_id} missing 'constraints' section"


def test_model_has_capabilities():
    """Verify models have required capability flags."""
    registry_path = (
        Path(__file__).parents[4]
        / "darts"
        / "models"
        / "forecasting"
        / "foundation"
        / "registry.yaml"
    )

    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    models = registry["models"]

    # Get first model to check structure
    first_model = next(iter(models.values()))
    capabilities = first_model["capabilities"]

    # Check for required capability flags
    required_capabilities = [
        "multivariate",
        "probabilistic",
    ]

    for capability in required_capabilities:
        assert capability in capabilities, \
            f"model missing required capability: {capability}"
        assert isinstance(capabilities[capability], bool), \
            f"capability '{capability}' must be boolean"
