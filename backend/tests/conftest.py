import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from satwiser_api import main  # noqa: E402
from satwiser_api.loader import load  # noqa: E402
from satwiser_api.repository import DatabaseRepository, LocalRepository  # noqa: E402

COUNTS = {"esa_manoeuvres": 2, "detected": 1, "missed": 1, "false_alarms": 1}


@pytest.fixture(scope="session")
def app_dir(tmp_path_factory) -> Path:
    """Minimal export with the same layout as ``scripts/export_app_data.py``."""
    root = tmp_path_factory.mktemp("app")
    (root / "lab").mkdir()
    sat = {"id": "S1A", "name": "Sentinel-1A", "first": "2022-12-31T00:00:00",
           "last": "2023-01-02T00:00:00", "operational_start": "2014-08-01",
           "split": "2020-01-01", "labels_end": "2023-01-02T00:00:00", "years": [2022, 2023],
           "default_lab_event": "S1A-D101",
           "summary": {"all": COUNTS, "by_split": {"test": COUNTS}, "by_year": {"2023": COUNTS}}}
    (root / "satellites.json").write_text(json.dumps([sat]))
    times = pd.date_range("2022-12-31", periods=40, freq="98min")
    pd.DataFrame({"satellite": "S1A", "orbit": np.arange(80, 120), "time": times,
                  "a_m": 7_071_000.0 + np.arange(40) * 0.1, "f107": 150.0, "ap": 8.0}
                 ).to_parquet(root / "series_S1A.parquet", index=False)
    pd.DataFrame({"satellite": "S1A", "date": pd.to_datetime(["2022-12-31", "2023-01-01"]),
                  "a_m": [7_071_000.0, 7_071_001.0], "f107": [150.0, np.nan], "ap": [8.0, 9.0]}
                 ).to_parquet(root / "daily_S1A.parquet", index=False)
    events = pd.DataFrame({
        "id": ["S1A-D101", "S1A-M105", "S1A-F110"], "satellite": "S1A",
        "kind": ["detected", "missed", "false_alarm"],
        "time": pd.to_datetime(["2023-01-01 02:00", "2023-01-01 09:00", "2023-01-02 01:00"]),
        "orbit": [101, 105, 110], "class": ["station_keeping", "orbit_change", "unexplained"],
        "esa_type": ["station_keeping", "inclination", None],
        "dv_est_mm_s": [9.1, np.nan, 1.2], "dv_esa_mm_s": [9.5, 4.0, np.nan],
        "esa_da_m": [17.9, 7.5, np.nan], "da_m": [17.2, np.nan, 2.3],
        "di_mdeg": [0.01, np.nan, 0.0], "de_1e6": [0.1, np.nan, 0.0],
        "statistic": [30.0, np.nan, 7.0], "channel": ["a", None, "a"],
        "alarm_delay_revs": [1.0, np.nan, 2.0], "split": "test",
        "lab_available": [True, False, False]})
    events.to_parquet(root / "events_S1A.parquet", index=False)
    grid = {"axes": {"sigma_m": [0.01, 1.0], "dv_cm_s": [0.1, 1.0], "points_per_day": [8640],
                     "rho": [0.0]},
            "p_detect": [[[[0.5]], [[1.0]]], [[[0.0]], [[0.9]]]],
            "p_detect_index_order": ["sigma_m", "dv_cm_s", "points_per_day", "rho"],
            "min_dv_90_cm_s": [[[0.2]], [[1.0]]],
            "presets": {"pod": {"label": "POD Copernicus", "sigma_m": 0.03,
                                "points_per_day": 8640, "rho": 0.0}},
            "detector": {"window_revs": 12, "threshold": 4.0}, "trials_per_cell": 60,
            "window_days": 10}
    (root / "robustness.json").write_text(json.dumps(grid))
    metrics = {"satellite": "S1A", "detector": {"kappa": 2.0}, "test": {"recall": 0.96},
               "comparison": [{"detector": "baseline", "test f1": 0.78}],
               "noise_m": {"raw": 4.5, "template": 0.2}, "source": "test"}
    (root / "metrics.json").write_text(json.dumps(metrics))
    window = {"event_id": "S1A-D101", "satellite": "S1A", "start": "2022-12-27T00:00:00",
              "step_s": 60.0, "event_time": "2023-01-01T02:00:00", "t_offset_s": [0.0, 60.0],
              "orbit": [30, 30], "states": {c: [1.0, 2.0] for c in
                                            ("rx", "ry", "rz", "vx", "vy", "vz")},
              "template_a": [0.0] * 175, "detector": {"window_revs": 12},
              "esa_manoeuvres": [{"start": "2023-01-01T02:00:00", "dv_t_mm_s": 9.5,
                                  "type": "station_keeping"}]}
    with gzip.open(root / "lab" / "S1A-D101.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(window, fh)
    return root


@pytest.fixture(scope="session", params=["local", "database"])
def client(request, app_dir, tmp_path_factory):
    if request.param == "local":
        repo = LocalRepository(app_dir)
    else:
        url = f"sqlite:///{tmp_path_factory.mktemp('db') / 'satwiser.db'}"
        load(app_dir, url, verbose=False)
        repo = DatabaseRepository(url)
    main.app.dependency_overrides[main.get_repository] = lambda: repo
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()
