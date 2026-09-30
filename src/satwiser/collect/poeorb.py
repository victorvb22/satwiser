"""Download Sentinel-1 precise orbit files (AUX_POEORB) from the public AWS bucket.

The ``s1-orbits`` bucket is readable anonymously over HTTPS, so no AWS account or SDK
is needed: the S3 ListObjectsV2 XML API is queried directly.

File names follow the ESA convention::

    S1A_OPER_AUX_POEORB_OPOD_<created>_V<validity start>_<validity stop>.EOF

Each daily file covers 26 h (22:59:42 on day D-1 to 00:59:42 on day D+1). When the
same validity window has been produced more than once, the latest creation wins.

The bucket is keyed by creation date, and some windows were reprocessed long after
the fact, so the full key list of a satellite is fetched once rather than guessing a
creation-date range.
"""

from __future__ import annotations

import gzip
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BUCKET_URL = "https://s1-orbits.s3.us-west-2.amazonaws.com"
PREFIX = "AUX_POEORB/"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_NAME_RE = re.compile(
    r"(?P<sat>S1[ABC])_OPER_AUX_POEORB_OPOD_(?P<created>\d{8}T\d{6})"
    r"_V(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.EOF$"
)


@dataclass(frozen=True)
class OrbitFileRef:
    key: str
    satellite: str
    created: datetime
    validity_start: datetime
    validity_stop: datetime
    size: int

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        return f"{BUCKET_URL}/{self.key}"


def parse_key(key: str, size: int = 0) -> OrbitFileRef | None:
    match = _NAME_RE.search(key)
    if match is None:
        return None
    fmt = "%Y%m%dT%H%M%S"
    return OrbitFileRef(
        key=key,
        satellite=match["sat"],
        created=datetime.strptime(match["created"], fmt),
        validity_start=datetime.strptime(match["start"], fmt),
        validity_stop=datetime.strptime(match["stop"], fmt),
        size=size,
    )


def list_all(satellite: str, session: requests.Session | None = None) -> list[OrbitFileRef]:
    """List every orbit file of ``satellite`` in the bucket (a few thousand keys)."""
    session = session or requests.Session()
    refs: list[OrbitFileRef] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": f"{PREFIX}{satellite}_", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        response = session.get(BUCKET_URL, params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall("s3:Contents", _S3_NS):
            ref = parse_key(item.findtext("s3:Key", "", _S3_NS),
                            int(item.findtext("s3:Size", "0", _S3_NS)))
            if ref is not None:
                refs.append(ref)
        token = root.findtext("s3:NextContinuationToken", None, _S3_NS)
        if not token:
            return refs


def select_window(refs: list[OrbitFileRef], start: date, end: date) -> list[OrbitFileRef]:
    """Latest production of each daily file needed to cover ``[start, end]`` (UTC days).

    The file for UTC day D has its validity starting at 22:59:42 on day D-1.
    """
    last = end - timedelta(days=1)
    keep = [r for r in refs if start - timedelta(days=1) <= r.validity_start.date() < last]
    return latest_per_validity(keep)


def latest_per_validity(refs: list[OrbitFileRef]) -> list[OrbitFileRef]:
    """Keep only the most recent production for each validity window."""
    best: dict[tuple[str, datetime], OrbitFileRef] = {}
    for ref in refs:
        key = (ref.satellite, ref.validity_start)
        if key not in best or ref.created > best[key].created:
            best[key] = ref
    return sorted(best.values(), key=lambda r: r.validity_start)


def local_path(ref: OrbitFileRef, dest: Path) -> Path | None:
    """Existing local copy of ``ref`` (plain or gzip-compressed), if complete."""
    plain = dest / ref.name
    if plain.exists() and (ref.size == 0 or plain.stat().st_size == ref.size):
        return plain
    packed = dest / (ref.name + ".gz")
    return packed if packed.exists() else None


def _fetch(ref: OrbitFileRef, dest: Path, session: requests.Session, compress: bool,
           attempts: int = 5) -> Path:
    path = dest / (ref.name + (".gz" if compress else ""))
    tmp = path.with_name(path.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with session.get(ref.url, stream=True, timeout=120) as response:
                response.raise_for_status()
                opener = gzip.open if compress else open
                with opener(tmp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.replace(path)
            return path
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download(refs: list[OrbitFileRef], dest: Path, compress: bool = True, workers: int = 8,
             verbose: bool = True) -> list[Path]:
    """Download missing files, gzip-compressed by default (about 8x smaller).

    Files already present (plain or compressed) are skipped, so an interrupted run can
    simply be restarted.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    todo = []
    for ref in refs:
        existing = local_path(ref, dest)
        if existing is not None:
            paths[ref.name] = existing
        else:
            todo.append(ref)
    local = threading.local()

    def job(ref: OrbitFileRef) -> tuple[str, Path | None]:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        try:
            return ref.name, _fetch(ref, dest, local.session, compress)
        except requests.RequestException as exc:
            print(f"FAILED {ref.name}: {exc}", flush=True)
            return ref.name, None

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (name, path) in enumerate(pool.map(job, todo), 1):
            if path is None:
                failed.append(name)
            else:
                paths[name] = path
            if verbose and (n % 50 == 0 or n == len(todo)):
                print(f"downloaded {n}/{len(todo)}", flush=True)
    if failed:
        print(f"{len(failed)} files failed; rerun the same command to retry them.")
    return [paths[r.name] for r in refs if r.name in paths]
