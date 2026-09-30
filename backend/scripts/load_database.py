"""Load the exported app data into the database given by ``--database-url`` or
``DATABASE_URL``. See :mod:`satwiser_api.loader`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satwiser_api.config import load_settings  # noqa: E402
from satwiser_api.loader import load  # noqa: E402


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=settings.database_url)
    parser.add_argument("--app-data", type=Path, default=settings.app_data)
    args = parser.parse_args()
    if not args.database_url:
        parser.error("no database URL (use --database-url or set DATABASE_URL)")
    load(args.app_data, args.database_url)


if __name__ == "__main__":
    main()
