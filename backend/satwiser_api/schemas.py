"""Response schemas (also the OpenAPI contract consumed by the front end)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: str
    backend: str
    attribution: str


class Counts(BaseModel):
    esa_manoeuvres: int
    detected: int
    missed: int
    false_alarms: int


class Summary(BaseModel):
    all: Counts
    by_split: dict[str, Counts]
    by_year: dict[str, Counts]


class Satellite(BaseModel):
    id: str
    name: str
    first: str
    last: str
    operational_start: str
    split: str
    labels_end: str
    years: list[int]
    default_lab_event: str
    summary: Summary


class Series(BaseModel):
    satellite: str
    year: int
    orbit: list[int]
    t_ms: list[int]
    a_m: list[float | None]
    f107: list[float | None]


class Overview(BaseModel):
    satellite: str
    t_ms: list[int]
    a_m: list[float | None]
    f107: list[float | None]


class EventSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    kind: str
    time: str
    orbit: int
    klass: str | None = Field(None, alias="class", serialization_alias="class")
    dv_est_mm_s: float | None
    dv_esa_mm_s: float | None
    da_m: float | None
    lab_available: bool


class Event(EventSummary):
    satellite: str
    esa_type: str | None
    esa_da_m: float | None
    di_mdeg: float | None
    de_1e6: float | None
    statistic: float | None
    channel: str | None
    alarm_delay_revs: float | None
    split: str | None


class Robustness(BaseModel):
    axes: dict[str, list[float]]
    p_detect: list
    p_detect_index_order: list[str]
    min_dv_90_cm_s: list
    presets: dict[str, dict]
    detector: dict
    trials_per_cell: int
    window_days: float


class LabManoeuvre(BaseModel):
    start: str
    dv_t_mm_s: float
    type: str


class LabWindow(BaseModel):
    event_id: str
    satellite: str
    start: str
    step_s: float
    event_time: str
    t_offset_s: list[float]
    orbit: list[int]
    states: dict[str, list[float]]
    template_a: list[float]
    detector: dict
    esa_manoeuvres: list[LabManoeuvre]


class Metrics(BaseModel):
    satellite: str
    detector: dict
    test: dict
    comparison: list[dict]
    noise_m: dict[str, float]
    model: dict = {}
    revolutions: int | None = None
    source: str
