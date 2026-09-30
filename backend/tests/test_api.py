"""Endpoint responses and schemas, run against both the file and the database backends."""


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "Copernicus Sentinel data" in body["attribution"]


def test_satellites(client):
    body = client.get("/api/satellites").json()
    assert [s["id"] for s in body] == ["S1A"]
    assert body[0]["summary"]["all"]["detected"] == 1
    assert client.get("/api/satellites/S1A").json()["name"] == "Sentinel-1A"
    assert client.get("/api/satellites/XXX").status_code == 404


def test_series_by_year(client):
    body = client.get("/api/satellites/S1A/series", params={"year": 2023}).json()
    assert body["year"] == 2023
    assert len(body["orbit"]) == len(body["t_ms"]) == len(body["a_m"]) > 0
    assert body["orbit"] == sorted(body["orbit"])
    assert body["t_ms"][0] >= 1672531200000  # 2023-01-01T00:00Z
    assert client.get("/api/satellites/S1A/series", params={"year": 2030}).status_code == 404
    assert client.get("/api/satellites/S1A/series").status_code == 422


def test_overview_handles_missing_values(client):
    body = client.get("/api/satellites/S1A/overview").json()
    assert body["f107"] == [150.0, None]


def test_events_filters(client):
    all_events = client.get("/api/satellites/S1A/events").json()
    assert {e["kind"] for e in all_events} == {"detected", "missed", "false_alarm"}
    assert all_events[0]["class"] == "station_keeping"
    missed = client.get("/api/satellites/S1A/events", params={"kind": "missed"}).json()
    assert [e["id"] for e in missed] == ["S1A-M105"]
    assert missed[0]["dv_est_mm_s"] is None
    assert client.get("/api/satellites/S1A/events", params={"year": 2022}).json() == []
    assert client.get("/api/satellites/S1A/events", params={"kind": "bogus"}).status_code == 422


def test_event_detail(client):
    body = client.get("/api/events/S1A-D101").json()
    assert body["dv_esa_mm_s"] == 9.5
    assert body["time"] == "2023-01-01T02:00:00Z"
    assert body["lab_available"] is True
    assert client.get("/api/events/S1A-D999").status_code == 404


def test_robustness(client):
    body = client.get("/api/robustness").json()
    assert body["p_detect_index_order"][0] == "sigma_m"
    assert body["presets"]["pod"]["label"] == "POD Copernicus"


def test_lab_window(client):
    response = client.get("/api/lab/S1A-D101")
    assert response.headers["content-encoding"] == "gzip"
    body = response.json()
    assert len(body["template_a"]) == 175
    assert set(body["states"]) == {"rx", "ry", "rz", "vx", "vy", "vz"}
    assert client.get("/api/lab/S1A-F110").status_code == 404


def test_lab_window_rejects_unexpected_ids(client):
    for bad in ("..%2Fsatellites", "S1A-D1.json", "x", "S1A-D101%2F..%2F..%2Fetc"):
        assert client.get(f"/api/lab/{bad}").status_code == 404


def test_read_only(client):
    assert client.post("/api/satellites").status_code == 405


def test_metrics(client):
    body = client.get("/api/metrics").json()
    assert body["test"]["recall"] == 0.96
    assert body["noise_m"]["template"] == 0.2
