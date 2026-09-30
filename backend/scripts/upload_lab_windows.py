"""Upload the lab windows to a public Supabase Storage bucket.

The windows (``$SATWISER_DATA_DIR/app/lab/*.json.gz``, about 0.5 MB each) are too large
for the free database quota; Supabase Storage has its own quota. The bucket is created
public-read if missing; writes need the service key, which stays on the workstation.

    python backend/scripts/upload_lab_windows.py

with ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY`` in the environment or the local
``.env`` (never committed).

Then set ``SATWISER_LAB_STORAGE_URL=<SUPABASE_URL>/storage/v1/object/public/lab-windows``
on the API.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satwiser_api.config import get_setting, load_settings  # noqa: E402

BUCKET = "lab-windows"


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--supabase-url", default=get_setting("SUPABASE_URL"))
    parser.add_argument("--service-key", default=get_setting("SUPABASE_SERVICE_KEY"))
    parser.add_argument("--app-data", type=Path, default=settings.app_data)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.supabase_url or not args.service_key:
        parser.error("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
    base = args.supabase_url.rstrip("/") + "/storage/v1"
    # New-style secret keys (sb_secret_...) are not JWTs: they go in the apikey header
    # only, and the gateway authorises the request as service_role. Legacy service_role
    # JWTs are also sent as bearer tokens.
    headers = {"apikey": args.service_key}
    if not args.service_key.startswith("sb_"):
        headers["Authorization"] = f"Bearer {args.service_key}"

    existing = requests.get(f"{base}/bucket/{BUCKET}", headers=headers, timeout=30)
    if existing.status_code != 200:
        created = requests.post(f"{base}/bucket", headers=headers, timeout=30,
                                json={"id": BUCKET, "name": BUCKET, "public": True})
        created.raise_for_status()
        print(f"created public bucket {BUCKET}")

    files = sorted((args.app_data / "lab").glob("*.json.gz"))

    def upload(path: Path) -> str | None:
        with requests.Session() as session:
            response = session.post(
                f"{base}/object/{BUCKET}/{path.name}", data=path.read_bytes(), timeout=120,
                headers={**headers, "Content-Type": "application/gzip", "x-upsert": "true"})
        return None if response.ok else f"{path.name}: {response.status_code} {response.text[:120]}"

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, error in enumerate(pool.map(upload, files), 1):
            if error:
                failures.append(error)
            if n % 100 == 0 or n == len(files):
                print(f"{n}/{len(files)} uploaded", flush=True)
    for failure in failures:
        print("FAILED", failure)
    print(f"public base URL: {args.supabase_url.rstrip('/')}/storage/v1/object/public/{BUCKET}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
