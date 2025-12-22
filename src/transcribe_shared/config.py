"""Generic configuration utilities for YAML loading and env overrides."""

from pathlib import Path
from typing import Any, Optional
import os
import re
import yaml


# Pattern for ${VAR:-default} or ${VAR} syntax
ENV_VAR_PATTERN = re.compile(r'\$\{([^}:\s]+)(?::-([^}]*))?\}')


def resolve_config_path(
    cli_path: Optional[Path],
    env_var: str,
    default_path: Path,
) -> Path:
    """
    Resolve config file path with priority: CLI > env var > default.

    Args:
        cli_path: Path from command-line argument (highest priority)
        env_var: Environment variable name to check (e.g., "TRANSCRIBE_M4_CONFIG")
        default_path: Default path if neither CLI nor env var set

    Returns:
        Resolved Path to config file
    """
    if cli_path is not None:
        return cli_path

    env_path = os.environ.get(env_var)
    if env_path:
        return Path(env_path)

    return default_path


def _interpolate_env_vars(value: Any) -> Any:
    """
    Recursively interpolate ${VAR:-default} patterns in config values.

    Supports:
        ${VAR} - replaced with env var value, or empty string if not set
        ${VAR:-default} - replaced with env var value, or default if not set
    """
    if isinstance(value, str):
        def replace_match(match):
            var_name = match.group(1)
            default = match.group(2)  # None if no default specified
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            return default if default is not None else ''

        result = ENV_VAR_PATTERN.sub(replace_match, value)
        # Convert "null" or "none" strings to actual None
        if result.lower() in ('null', 'none', ''):
            return None
        return result
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def load_yaml(path: Path) -> dict:
    """
    Load configuration from YAML file with environment variable interpolation.

    Supports ${VAR:-default} syntax in string values:
        ${VAR} - replaced with env var value, or empty string if not set
        ${VAR:-default} - replaced with env var value, or default if not set

    Args:
        path: Path to YAML file

    Returns:
        Dict of configuration values, empty dict if file doesn't exist
    """
    if not path.exists():
        return {}
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    return _interpolate_env_vars(config)


def apply_env_overrides(config: dict, dataclass_type: type) -> dict:
    """
    Apply environment variable overrides using uppercased yaml key names.

    E.g., huggingface_token -> HUGGINGFACE_TOKEN

    Args:
        config: Base configuration dict
        dataclass_type: Dataclass to get field types for conversion

    Returns:
        Updated configuration dict
    """
    from dataclasses import fields as dataclass_fields

    result = config.copy()

    # Get field info for type conversion
    field_types = {f.name: f.type for f in dataclass_fields(dataclass_type)}

    for key, field_type in field_types.items():
        env_var = key.upper()
        value = os.environ.get(env_var)
        if value is not None:
            # Handle null/none values
            if value.lower() in ("null", "none", ""):
                result[key] = None
            else:
                # Type conversion based on dataclass field type
                origin = getattr(field_type, '__origin__', None)
                args = getattr(field_type, '__args__', ())

                if field_type == int:
                    result[key] = int(value)
                elif field_type == float:
                    result[key] = float(value)
                elif field_type == bool:
                    result[key] = value.lower() in ("true", "1", "yes")
                elif int in args:
                    result[key] = int(value)
                elif float in args:
                    result[key] = float(value)
                else:
                    result[key] = value

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
