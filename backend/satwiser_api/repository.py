"""Read-only data access: the same interface over exported files or a SQL database.

Both implementations return plain JSON-ready structures (NaN becomes ``None``, times are
UTC epoch milliseconds or ISO strings with a ``Z`` suffix), so the API layer does not
depend on the storage.
"""

from __future__ import annotations

import gzip
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select, text

from satwiser_api import db

SUMMARY_FIELDS = ("id", "kind", "time", "orbit", "class", "dv_est_mm_s", "dv_esa_mm_s",
                  "da_m", "lab_available")
KINDS = ("detected", "missed", "false_alarm", "unscored")


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        return _clean(value.item())
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
    return value


def _epoch_ms(value: datetime) -> int:
    return int(round((value.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds() * 1e3))


def _round(value, digits: int):
    value = _clean(value)
    return None if value is None else round(float(value), digits)


class Repository(Protocol):
    def check(self) -> None: ...
    def satellites(self) -> list[dict]: ...
    def series(self, satellite: str, year: int) -> dict | None: ...
    def overview(self, satellite: str) -> dict | None: ...
    def events(self, satellite: str, year: int | None, kind: str | None) -> list[dict]: ...
    def event(self, event_id: str) -> dict | None: ...
    def robustness(self) -> dict | None: ...
    def metrics(self) -> dict | None: ...
    def lab_window(self, event_id: str) -> dict | None: ...


def _series_payload(satellite: str, year: int, rows) -> dict:
    return {
        "satellite": satellite, "year": year,
        "orbit": [int(r["orbit"]) for r in rows],
        "t_ms": [_epoch_ms(r["time"]) for r in rows],
        "a_m": [_round(r["a_m"], 3) for r in rows],
        "f107": [_round(r["f107"], 1) for r in rows],
    }


class LocalRepository:
    """Files written by ``scripts/export_app_data.py`` (loaded once, kept in memory)."""

    def __init__(self, folder: Path):
        import pandas as pd

        self.folder = Path(folder)
        self._satellites = json.loads((self.folder / "satellites.json").read_text())
        self._series, self._daily, self._events = {}, {}, {}
        for sat in (s["id"] for s in self._satellites):
            series = pd.read_parquet(self.folder / f"series_{sat}.parquet")
            self._series[sat] = series.assign(year=series["time"].dt.year)
            self._daily[sat] = pd.read_parquet(self.folder / f"daily_{sat}.parquet")
            self._events[sat] = pd.read_parquet(self.folder / f"events_{sat}.parquet")
        self._documents = {}
        for key in ("robustness", "metrics"):
            path = self.folder / f"{key}.json"
            self._documents[key] = json.loads(path.read_text()) if path.exists() else None

    def check(self) -> None:
        return None

    def satellites(self) -> list[dict]:
        return self._satellites

    def series(self, satellite: str, year: int) -> dict | None:
        frame = self._series.get(satellite)
        if frame is None:
            return None
        rows = frame[frame["year"] == year].sort_values("orbit")
        if rows.empty:
            return None
        return _series_payload(satellite, year, rows.to_dict("records"))

    def overview(self, satellite: str) -> dict | None:
        frame = self._daily.get(satellite)
        if frame is None:
            return None
        return {"satellite": satellite,
                "t_ms": [_epoch_ms(t) for t in frame["date"]],
                "a_m": [_round(v, 2) for v in frame["a_m"]],
                "f107": [_round(v, 1) for v in frame["f107"]]}

    def events(self, satellite: str, year: int | None, kind: str | None) -> list[dict]:
        frame = self._events.get(satellite)
        if frame is None:
            return []
        if year is not None:
            frame = frame[frame["time"].dt.year == year]
        if kind is not None:
            frame = frame[frame["kind"] == kind]
        return [{k: _clean(r[k]) for k in SUMMARY_FIELDS} for r in frame.to_dict("records")]

    def event(self, event_id: str) -> dict | None:
        satellite = event_id.split("-")[0]
        frame = self._events.get(satellite)
        if frame is None:
            return None
        rows = frame[frame["id"] == event_id]
        if rows.empty:
            return None
        return {k: _clean(v) for k, v in rows.iloc[0].to_dict().items()}

    def robustness(self) -> dict | None:
        return self._documents["robustness"]

    def metrics(self) -> dict | None:
        return self._documents["metrics"]

    def lab_window(self, event_id: str) -> dict | None:
        path = self.folder / "lab" / f"{Path(event_id).name}.json.gz"
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)


class DatabaseRepository:
    """Tables created and filled by ``backend/scripts/load_database.py``."""

    def __init__(self, url: str):
        self.engine = db.make_engine(url)

    def check(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def satellites(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(db.satellites.c.payload).order_by(db.satellites.c.id))
            return [json.loads(r.payload) for r in rows]

    def series(self, satellite: str, year: int) -> dict | None:
        t = db.revolutions
        start, stop = datetime(year, 1, 1), datetime(year + 1, 1, 1)
        with self.engine.connect() as conn:
            rows = conn.execute(select(t).where(t.c.satellite == satellite, t.c.time >= start,
                                                t.c.time < stop).order_by(t.c.orbit))
            rows = [r._mapping for r in rows]
        return _series_payload(satellite, year, rows) if rows else None

    def overview(self, satellite: str) -> dict | None:
        t = db.daily
        with self.engine.connect() as conn:
            rows = [r._mapping for r in conn.execute(
                select(t).where(t.c.satellite == satellite).order_by(t.c.date))]
        if not rows:
            return None
        return {"satellite": satellite, "t_ms": [_epoch_ms(r["date"]) for r in rows],
                "a_m": [_round(r["a_m"], 2) for r in rows],
                "f107": [_round(r["f107"], 1) for r in rows]}

    def events(self, satellite: str, year: int | None, kind: str | None) -> list[dict]:
        t = db.events
        query = select(*[t.c[k] for k in SUMMARY_FIELDS]).where(t.c.satellite == satellite)
        if year is not None:
            query = query.where(t.c.time >= datetime(year, 1, 1),
                                t.c.time < datetime(year + 1, 1, 1))
        if kind is not None:
            query = query.where(t.c.kind == kind)
        with self.engine.connect() as conn:
            rows = conn.execute(query.order_by(t.c.time))
            return [{k: _clean(v) for k, v in r._mapping.items()} for r in rows]

    def event(self, event_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(db.events).where(db.events.c.id == event_id)).first()
        return None if row is None else {k: _clean(v) for k, v in row._mapping.items()}

    def _document(self, key: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(db.documents.c.payload)
                               .where(db.documents.c.key == key)).first()
        return None if row is None else json.loads(row.payload)

    def robustness(self) -> dict | None:
        return self._document("robustness")

    def metrics(self) -> dict | None:
        return self._document("metrics")

    def lab_window(self, event_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(db.lab_windows.c.payload_gz)
                               .where(db.lab_windows.c.event_id == event_id)).first()
        return None if row is None else json.loads(gzip.decompress(row.payload_gz))
