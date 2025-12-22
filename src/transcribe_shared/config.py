"""Generic configuration utilities for YAML loading and env overrides."""

from pathlib import Path
from typing import Any
import os
import yaml


def load_yaml(path: Path) -> dict:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Dict of configuration values, empty dict if file doesn't exist
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_env_overrides(
    config: dict,
    env_mapping: dict[str, str],
    type_conversions: dict[str, type] | None = None,
) -> dict:
    """
    Apply environment variable overrides to config.

    Args:
        config: Base configuration dict
        env_mapping: Map of ENV_VAR_NAME -> config_key
        type_conversions: Map of config_key -> type (int, float, bool)

    Returns:
        Updated configuration dict
    """
    if type_conversions is None:
        type_conversions = {}

    result = config.copy()

    for env_var, config_key in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Handle null/none values
            if value.lower() in ("null", "none"):
                result[config_key] = None
            # Apply type conversion if specified
            elif config_key in type_conversions:
                target_type = type_conversions[config_key]
                if target_type == bool:
                    result[config_key] = value.lower() in ("true", "1", "yes")
                else:
                    result[config_key] = target_type(value)
            else:
                result[config_key] = value

    return result


def filter_to_dataclass(config: dict, dataclass_type: type) -> dict:
    """
    Filter config dict to only include keys valid for a dataclass.

    Args:
        config: Configuration dict
        dataclass_type: Dataclass to filter for

    Returns:
        Config dict with only valid keys
    """
    valid_keys = {f.name for f in dataclass_type.__dataclass_fields__.values()}
    return {k: v for k, v in config.items() if k in valid_keys}
