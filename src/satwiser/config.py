"""Project-wide paths and constants."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Root folder for raw downloads and intermediate products.

    Resolved from the ``SATWISER_DATA_DIR`` environment variable (or a ``.env`` file at
    the repository root), falling back to ``<repo>/data`` which is git-ignored.
    """
    value = os.environ.get("SATWISER_DATA_DIR") or _read_dotenv().get("SATWISER_DATA_DIR")
    return Path(value) if value else REPO_ROOT / "data"


def raw_dir(*parts: str) -> Path:
    path = data_dir().joinpath("raw", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(*parts: str) -> Path:
    path = data_dir().joinpath("processed", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_dotenv() -> dict[str, str]:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values
