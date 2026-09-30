"""Runtime settings, read from environment variables (and a local ``.env`` if present).

- ``SATWISER_DATA_BACKEND``: ``local`` (files exported by the pipeline) or ``database``;
- ``SATWISER_APP_DATA``: folder of the exported files (default
  ``$SATWISER_DATA_DIR/app``);
- ``DATABASE_URL``: SQLAlchemy URL of the Postgres database (Supabase in production);
- ``SATWISER_LAB_STORAGE_URL``: public base URL of the object storage holding the lab
  windows (``<base>/<event id>.json.gz``), used with the database backend;
- ``CORS_ORIGINS``: comma-separated list of allowed front-end origins.

No secret is ever sent to the browser: the database URL stays on the server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_file in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    values.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    return values


def _get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _dotenv().get(name) or default


@dataclass(frozen=True)
class Settings:
    backend: str
    app_data: Path
    database_url: str | None
    lab_storage_url: str | None
    cors_origins: tuple[str, ...]


def load_settings() -> Settings:
    data_dir = _get("SATWISER_DATA_DIR", str(REPO_ROOT / "data"))
    app_data = Path(_get("SATWISER_APP_DATA", str(Path(data_dir) / "app")))
    origins = _get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return Settings(
        backend=_get("SATWISER_DATA_BACKEND", "local"),
        app_data=app_data,
        database_url=_get("DATABASE_URL"),
        lab_storage_url=_get("SATWISER_LAB_STORAGE_URL"),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )
