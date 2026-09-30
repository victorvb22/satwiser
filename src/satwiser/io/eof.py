"""Parse Sentinel-1 AUX_POEORB Earth Explorer files (XML ``.EOF``).

State vectors are given every 10 s in an Earth-fixed frame (ITRF), with UTC, TAI and UT1
time tags. Consecutive daily files overlap by about two hours; :func:`merge` removes the
duplicates and reports gaps.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

_FIELDS = ("X", "Y", "Z", "VX", "VY", "VZ")


def read_eof(path: Path | str) -> pd.DataFrame:
    """Read one orbit file.

    Returns a frame indexed by UTC timestamp with columns ``x, y, z`` (m), ``vx, vy, vz``
    (m/s) in the Earth-fixed frame, ``ut1_utc`` (s), ``orbit`` (absolute orbit number)
    and ``quality``.
    """
    tree = etree.parse(str(path))
    osvs = tree.findall(".//List_of_OSVs/OSV")
    n = len(osvs)
    utc = np.empty(n, dtype="datetime64[us]")
    ut1 = np.empty(n, dtype="datetime64[us]")
    values = np.empty((n, 6))
    orbit = np.empty(n, dtype=np.int64)
    quality = np.empty(n, dtype=object)
    for i, osv in enumerate(osvs):
        utc[i] = np.datetime64(osv.findtext("UTC")[4:])
        ut1[i] = np.datetime64(osv.findtext("UT1")[4:])
        orbit[i] = int(osv.findtext("Absolute_Orbit"))
        quality[i] = osv.findtext("Quality")
        for j, field in enumerate(_FIELDS):
            values[i, j] = float(osv.findtext(field))
    frame = pd.DataFrame(values, columns=[f.lower() for f in _FIELDS])
    frame["ut1_utc"] = (ut1 - utc).astype("timedelta64[us]").astype(np.int64) * 1e-6
    frame["orbit"] = orbit
    frame["quality"] = quality
    frame.index = pd.DatetimeIndex(utc, name="utc").as_unit("ns")
    return frame


def merge(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate daily files, keeping the first occurrence of each epoch.

    Files are expected in chronological order; for a duplicated epoch the value from the
    earlier file is kept (the later file only extends coverage).
    """
    merged = pd.concat(list(frames))
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged.sort_index()


def find_gaps(index: pd.DatetimeIndex, nominal_step_s: float = 10.0,
              tolerance: float = 1.5) -> pd.DataFrame:
    """List intervals where consecutive epochs are further apart than expected."""
    dt = np.diff(index.as_unit("ns").asi8) * 1e-9
    mask = dt > nominal_step_s * tolerance
    return pd.DataFrame(
        {
            "start": index[:-1][mask],
            "stop": index[1:][mask],
            "duration_s": dt[mask],
        }
    )
