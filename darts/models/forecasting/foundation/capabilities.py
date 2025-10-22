"""Foundation model capabilities registry loader.

This module provides functions to load and query the capabilities.yaml registry
which defines what each foundation model family/subfamily/variant can do.
"""
from pathlib import Path
from typing import Any, Dict

import yaml


def load_capabilities() -> Dict[str, Any]:
    """Load capabilities registry from capabilities.yaml.

    Returns:
        Dictionary containing the complete capabilities registry.

    Raises:
        FileNotFoundError: If capabilities.yaml doesn't exist.
        yaml.YAMLError: If capabilities.yaml is invalid.
    """
    capabilities_path = Path(__file__).parent / "capabilities.yaml"

    with open(capabilities_path, "r") as f:
        return yaml.safe_load(f)


def get_family(family_name: str) -> Dict[str, Any]:
    """Get capabilities for a specific model family.

    Args:
        family_name: Name of the model family (e.g., "chronos", "timesfm").

    Returns:
        Dictionary containing family capabilities including subfamilies.

    Raises:
        KeyError: If family doesn't exist in registry.
    """
    capabilities = load_capabilities()

    if family_name not in capabilities:
        raise KeyError(f"Family '{family_name}' not found in capabilities registry")

    return capabilities[family_name]


def get_variant(
    family_name: str, subfamily_name: str, variant_name: str
) -> Dict[str, Any]:
    """Get capabilities for a specific model variant.

    Args:
        family_name: Name of the model family (e.g., "chronos").
        subfamily_name: Name of the subfamily (e.g., "chronos-2").
        variant_name: Name of the variant (e.g., "base", "small", "large").

    Returns:
        Dictionary containing variant capabilities (multivariate, probabilistic, etc.).

    Raises:
        KeyError: If family, subfamily, or variant doesn't exist.
    """
    family = get_family(family_name)

    if "subfamilies" not in family:
        raise KeyError(
            f"Family '{family_name}' has no subfamilies"
        )

    subfamilies = family["subfamilies"]
    if subfamily_name not in subfamilies:
        raise KeyError(
            f"Subfamily '{subfamily_name}' not found in family '{family_name}'"
        )

    subfamily = subfamilies[subfamily_name]
    if "variants" not in subfamily:
        raise KeyError(
            f"Subfamily '{subfamily_name}' has no variants"
        )

    variants = subfamily["variants"]
    if variant_name not in variants:
        raise KeyError(
            f"Variant '{variant_name}' not found in subfamily '{subfamily_name}'"
        )

    return variants[variant_name]
