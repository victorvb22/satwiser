"""Parse the Sentinel-1 flight dynamics history files published on SentiWiki.

Three plain-text formats, each starting with a one-line header (generation time and an
integer identifier):

``.man`` (manoeuvre history)
    Burns as pairs of lines: ``<UTC> <a1> <a2> <a3> <flag>`` with ``flag = 1`` at burn
    start and ``0`` at burn end. ``a1..a3`` are the three components of the commanded
    thrust acceleration written with Fortran ``D`` exponents. The file carries no unit or
    frame; the step 1 reconnaissance checks them against the orbit (see
    ``scripts/recon_step1.py``).
``.out`` (outages)
    ``<UTC start> <UTC stop> <MAN|GAP>``: manoeuvre windows and GNSS data gaps.
``.mhf`` (mass history)
    ``<Y> <M> <D> <h> <m> <s> <mass kg> <cog x> <cog y> <cog z>``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_TIME_FMT = "%Y/%m/%d-%H:%M:%S.%f"
_FORTRAN_RE = re.compile(r"[-+]?\d\.\d+[DdEe][-+]\d+")


def _fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _split_man_line(line: str) -> tuple[str, list[float], int]:
    """Split a ``.man`` record.

    Components are packed without separating blanks when negative
    (``-0.4D-09-0.4D-06``), so they are matched by pattern rather than split on spaces.
    """
    stamp, body = line[:23], line[23:]
    numbers = _FORTRAN_RE.findall(body)
    if len(numbers) != 3:
        raise ValueError(f"Malformed manoeuvre record: {line!r}")
    flag = int(body.split()[-1])
    return stamp, [_fortran_float(n) for n in numbers], flag


def read_manoeuvres(path: Path | str) -> pd.DataFrame:
    """One row per burn: ``start, stop, duration_s, a1, a2, a3, a_norm``.

    Acceleration components are returned exactly as written in the file (no unit
    conversion).
    """
    lines = [ln for ln in Path(path).read_text().splitlines()[1:] if ln.strip()]
    rows = []
    pending: tuple[pd.Timestamp, list[float]] | None = None
    for line in lines:
        stamp, acc, flag = _split_man_line(line)
        t = pd.to_datetime(stamp, format=_TIME_FMT)
        if flag == 1:
            if pending is not None:
                raise ValueError(f"Burn starting at {pending[0]} has no end record")
            pending = (t, acc)
        else:
            if pending is None:
                raise ValueError(f"Burn end at {t} without a start record")
            start, acc_start = pending
            rows.append((start, t, *acc_start))
            pending = None
    if pending is not None:
        raise ValueError(f"Burn starting at {pending[0]} has no end record")
    frame = pd.DataFrame(rows, columns=["start", "stop", "a1", "a2", "a3"])
    frame.insert(2, "duration_s", (frame["stop"] - frame["start"]).dt.total_seconds())
    frame["a_norm"] = np.sqrt(frame["a1"] ** 2 + frame["a2"] ** 2 + frame["a3"] ** 2)
    return frame


def read_outages(path: Path | str) -> pd.DataFrame:
    """One row per outage: ``start, stop, kind`` with ``kind`` in ``{"MAN", "GAP"}``."""
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 3:
            rows.append(parts)
    frame = pd.DataFrame(rows, columns=["start", "stop", "kind"])
    for column in ("start", "stop"):
        frame[column] = pd.to_datetime(frame[column], format=_TIME_FMT)
    return frame


def read_mass_history(path: Path | str) -> pd.DataFrame:
    """Mass (kg) and centre of gravity (m) indexed by UTC time."""
    cols = ["year", "month", "day", "hour", "minute", "second", "mass_kg",
            "cog_x", "cog_y", "cog_z"]
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    seconds = raw.pop("second")
    stamp = pd.to_datetime(raw[["year", "month", "day", "hour", "minute"]])
    stamp = stamp + pd.to_timedelta(seconds, unit="s")
    return raw[["mass_kg", "cog_x", "cog_y", "cog_z"]].set_index(stamp.rename("utc"))
