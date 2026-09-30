"""Database schema (SQLAlchemy Core), shared by the API and the import script.

Portable types only, so the same schema runs on Postgres (Supabase) and SQLite (tests).
JSON documents are stored as text. Lab windows (about 0.5 MB each) are not in the
database: they live in object storage (Supabase Storage) or in the local export.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)

metadata = MetaData()

satellites = Table(
    "satellites", metadata,
    Column("id", String(8), primary_key=True),
    Column("payload", Text, nullable=False),
)

revolutions = Table(
    "revolutions", metadata,
    Column("satellite", String(8), primary_key=True),
    Column("orbit", Integer, primary_key=True),
    Column("time", DateTime, nullable=False),
    Column("a_m", Float, nullable=False),
    Column("f107", Float),
    Column("ap", Float),
    Index("ix_revolutions_sat_time", "satellite", "time"),
)

daily = Table(
    "daily", metadata,
    Column("satellite", String(8), primary_key=True),
    Column("date", DateTime, primary_key=True),
    Column("a_m", Float),
    Column("f107", Float),
    Column("ap", Float),
)

events = Table(
    "events", metadata,
    Column("id", String(32), primary_key=True),
    Column("satellite", String(8), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("time", DateTime, nullable=False),
    Column("orbit", Integer, nullable=False),
    Column("class", String(24)),
    Column("esa_type", String(24)),
    Column("dv_est_mm_s", Float),
    Column("dv_esa_mm_s", Float),
    Column("esa_da_m", Float),
    Column("da_m", Float),
    Column("di_mdeg", Float),
    Column("de_1e6", Float),
    Column("statistic", Float),
    Column("channel", String(4)),
    Column("alarm_delay_revs", Float),
    Column("split", String(16)),
    Column("lab_available", Boolean, nullable=False, default=False),
    Index("ix_events_sat_time", "satellite", "time"),
)

documents = Table(
    "documents", metadata,
    Column("key", String(64), primary_key=True),
    Column("payload", Text, nullable=False),
)

EVENT_COLUMNS = [c.name for c in events.columns]


def make_engine(url: str) -> Engine:
    """Engine with settings suited to Supabase's connection pooler.

    The transaction-mode pooler does not support server-side prepared statements, so
    psycopg's automatic preparation is disabled for Postgres URLs.
    """
    connect_args = {"prepare_threshold": None} if url.startswith("postgresql+psycopg") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)
