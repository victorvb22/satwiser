"""Build per-revolution mean element series from a directory of POEORB files.

Files are parsed in parallel; each one contributes only the epochs before the validity
start of the next file (earlier file wins on the two-hour overlap), so contributions
are disjoint. Revolution means are then computed as a stream: samples after the last
ascending node crossing of a chunk are carried over to the next one, so revolutions
spanning file boundaries are averaged exactly once and memory stays bounded.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from satwiser.io.eof import read_eof
from satwiser.orbit.elements import elements_from_eof
from satwiser.orbit.revolutions import revolution_ids, revolution_means

_COLUMNS = ["a_nosp", "e", "i", "u", "orbit", "pod_flag"]


def validity_start(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.name.split("_V")[1][:15])


def orbit_files(folder: Path, satellite: str) -> list[Path]:
    """Orbit files of ``satellite`` (plain or gzip), one per validity start, sorted."""
    files: dict[pd.Timestamp, Path] = {}
    for path in folder.glob(f"{satellite}_OPER_AUX_POEORB_OPOD_*.EOF*"):
        if path.name.endswith(".part"):
            continue
        key = validity_start(path)
        # Several productions of the same window: keep the most recent creation.
        if key not in files or path.name > files[key].name:
            files[key] = path
    return [files[k] for k in sorted(files)]


def file_elements(path: Path, until: pd.Timestamp | None) -> pd.DataFrame:
    """Elements for the epochs of ``path`` before ``until`` (next file's validity start)."""
    raw = read_eof(path)
    if until is not None:
        raw = raw[raw.index < until]
    el = elements_from_eof(raw)
    el["pod_flag"] = (raw["quality"] != "NOMINAL").to_numpy()
    return el[_COLUMNS]


def _job(args: tuple[Path, pd.Timestamp | None]) -> pd.DataFrame:
    return file_elements(*args)


def iter_elements(files: list[Path], workers: int = 6) -> Iterator[pd.DataFrame]:
    """Yield disjoint per-file element frames in chronological order."""
    untils = [validity_start(p) for p in files[1:]] + [None]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_job, zip(files, untils, strict=True), chunksize=4)


def stream_revolutions(frames: Iterator[pd.DataFrame], chunk_files: int = 30,
                       verbose: bool = True) -> pd.DataFrame:
    """Per-revolution means over a chronological stream of element frames."""
    carry: pd.DataFrame | None = None
    pending: list[pd.DataFrame] = []
    results: list[pd.DataFrame] = []
    count = 0

    def flush(final: bool) -> None:
        nonlocal carry, pending
        parts = ([carry] if carry is not None else []) + pending
        pending = []
        if not parts:
            return
        buf = pd.concat(parts)
        buf = buf[~buf.index.duplicated(keep="first")].sort_index()
        if final:
            done, carry = buf, None
        else:
            ids = revolution_ids(buf["u"].to_numpy())
            last = ids == ids[-1]
            done, carry = buf[~last], buf[last]
        if len(done):
            results.append(revolution_means(done, trim_edges=False))

    for frame in frames:
        pending.append(frame)
        count += 1
        if len(pending) >= chunk_files:
            flush(final=False)
            if verbose:
                print(f"{count} files, {sum(len(r) for r in results)} revolutions", flush=True)
    flush(final=True)
    out = pd.concat(results)
    return out[~out.index.duplicated(keep="first")]


def add_space_weather(revs: pd.DataFrame, sw: pd.DataFrame) -> pd.DataFrame:
    """Join daily F10.7 (observed), its 81-day centred mean, and Ap by UTC date."""
    daily = sw[["f107_obs", "ap"]].copy()
    daily["f107_81d"] = daily["f107_obs"].rolling(81, center=True, min_periods=40).mean()
    day = revs.index.normalize()
    joined = daily.reindex(day)
    out = revs.copy()
    for col in ("f107_obs", "f107_81d", "ap"):
        out[col] = joined[col].to_numpy()
    return out


def build(folder: Path, satellite: str, workers: int = 6) -> pd.DataFrame:
    files = orbit_files(folder, satellite)
    if not files:
        raise FileNotFoundError(f"No {satellite} orbit files in {folder}")
    revs = stream_revolutions(iter_elements(files, workers))
    revs.insert(0, "satellite", satellite)
    return revs


def mean_period_s(revs: pd.DataFrame) -> float:
    return float(np.median((revs["t_stop"] - revs["t_start"]).dt.total_seconds()) + 10.0)
