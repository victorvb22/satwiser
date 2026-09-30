"""Record the external inputs behind the reports (``reports/inputs.json``).

The GFZ index file is updated daily (preliminary values become definitive) and the
orbit bucket keeps the latest production of each day, so rerunning the collection later
can change the inputs. This manifest pins what the committed reports were computed from:
sha256 of the ESA history files and of the GFZ file, and the list of orbit files used.

    python scripts/write_manifest.py --satellite S1A
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from satwiser.collect import spaceweather
from satwiser.config import REPO_ROOT, raw_dir
from satwiser.pipeline.series import orbit_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--satellite", default="S1A")
    sat = parser.parse_args().satellite
    orbits = [p.name.removesuffix(".gz") for p in orbit_files(raw_dir("poeorb", sat), sat)]
    manifest = {
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "esa": {p.name: sha256(p) for p in sorted(raw_dir("esa").glob("*.*"))},
        "spaceweather": {spaceweather.FILENAME:
                         sha256(raw_dir("spaceweather") / spaceweather.FILENAME)},
        "poeorb": {"satellite": sat, "files": len(orbits), "first": orbits[0],
                   "last": orbits[-1],
                   "names_sha256": hashlib.sha256("\n".join(orbits).encode()).hexdigest()},
    }
    out = REPO_ROOT / "reports" / "inputs.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
