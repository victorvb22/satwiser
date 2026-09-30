"""Load the exported app files into a SQL database (Supabase Postgres or SQLite).

Idempotent: rows of the satellites present in the export are replaced. Run from a
workstation with the pipeline outputs; the database URL is a server-side secret. Lab
windows go to object storage instead (``backend/scripts/upload_lab_windows.py``).

    python backend/scripts/load_database.py --database-url postgresql+psycopg://...
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import delete, insert, text

from satwiser_api import db

CHUNK = 5000


def _records(frame, columns):
    import pandas as pd

    out = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    records = out.to_dict("records")
    for rec in records:
        for key, value in rec.items():
            if hasattr(value, "to_pydatetime"):
                rec[key] = value.to_pydatetime(warn=False).replace(tzinfo=None)
    return records


def load(app_dir: Path, database_url: str, verbose: bool = True) -> dict[str, int]:
    import pandas as pd

    app_dir = Path(app_dir)
    engine = db.make_engine(database_url)
    db.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        # Supabase exposes the public schema through its REST API with the anonymous key:
        # row level security without any policy closes that door. The API connects as
        # the table owner, which bypasses RLS.
        with engine.begin() as conn:
            for table in db.metadata.sorted_tables:
                conn.execute(text(f'ALTER TABLE "{table.name}" ENABLE ROW LEVEL SECURITY'))
    sats = json.loads((app_dir / "satellites.json").read_text())
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for sat in sats:
            sid = sat["id"]
            for table in (db.revolutions, db.daily, db.events):
                conn.execute(delete(table).where(table.c.satellite == sid))
            conn.execute(delete(db.satellites).where(db.satellites.c.id == sid))
            conn.execute(insert(db.satellites), [{"id": sid, "payload": json.dumps(sat)}])

            series = pd.read_parquet(app_dir / f"series_{sid}.parquet")
            rows = _records(series, ["satellite", "orbit", "time", "a_m", "f107", "ap"])
            for k in range(0, len(rows), CHUNK):
                conn.execute(insert(db.revolutions), rows[k:k + CHUNK])
            counts[f"revolutions_{sid}"] = len(rows)

            daily = pd.read_parquet(app_dir / f"daily_{sid}.parquet")
            conn.execute(insert(db.daily), _records(daily, ["satellite", "date", "a_m",
                                                             "f107", "ap"]))
            counts[f"daily_{sid}"] = len(daily)

            events = pd.read_parquet(app_dir / f"events_{sid}.parquet")
            conn.execute(insert(db.events), _records(events, db.EVENT_COLUMNS))
            counts[f"events_{sid}"] = len(events)


        for key in ("robustness", "metrics"):
            path = app_dir / f"{key}.json"
            if path.exists():
                conn.execute(delete(db.documents).where(db.documents.c.key == key))
                conn.execute(insert(db.documents), [{"key": key, "payload": path.read_text()}])
    if verbose:
        print(json.dumps(counts, indent=2))
    return counts
