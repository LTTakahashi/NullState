"""Configuration loading utilities.

All pipeline parameters live in YAML config files, never hardcoded.
This module provides the shared loading and validation pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    ValueError
        If the parsed content is not a dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping, got {type(config).__name__}"
        )

    logger.info("Loaded config from %s (%d top-level keys)", config_path, len(config))
    return config


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value from a config dict.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    *keys : str
        Sequence of keys to traverse (e.g. ``"dish_vector", "cosine_threshold"``).
    default : Any
        Value to return if the key path does not exist.

    Returns
    -------
    Any
        The value at the specified path, or ``default``.
    """
    current = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current
