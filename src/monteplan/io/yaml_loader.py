"""YAML data file loader for tax brackets, RMD tables, etc."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> Any:
    """Load and parse a YAML file.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed YAML content (typically a dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_package_yaml(relative_path: str) -> Any:
    """Load a YAML file relative to the monteplan package root.

    Args:
        relative_path: Path relative to ``src/monteplan/``,
            e.g. ``"taxes/tables/us_federal_2026.yaml"``.

    Returns:
        Parsed YAML content.
    """
    package_root = Path(__file__).resolve().parent.parent
    return load_yaml(package_root / relative_path)
