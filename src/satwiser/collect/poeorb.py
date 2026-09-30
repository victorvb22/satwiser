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

import re
import xml.etree.ElementTree as ET
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


def download(refs: list[OrbitFileRef], dest: Path, session: requests.Session | None = None,
             verbose: bool = True) -> list[Path]:
    """Download files that are not already present with the expected size."""
    session = session or requests.Session()
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ref in enumerate(refs, 1):
        path = dest / ref.name
        if not (path.exists() and (ref.size == 0 or path.stat().st_size == ref.size)):
            tmp = path.with_suffix(".part")
            with session.get(ref.url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.replace(path)
            if verbose:
                print(f"[{i}/{len(refs)}] {ref.name}")
        paths.append(path)
    return paths
