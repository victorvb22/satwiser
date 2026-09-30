from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from satwiser.collect import poeorb, spaceweather
from satwiser.io import eof, esa_history

FIXTURES = Path(__file__).parent / "fixtures"


def test_read_eof_columns_and_times():
    frame = eof.read_eof(FIXTURES / "eof_a.EOF")
    assert list(frame.columns[:6]) == ["x", "y", "z", "vx", "vy", "vz"]
    assert len(frame) == 5
    assert frame.index[0] == pd.Timestamp("2022-12-11 22:59:42")
    assert np.all(np.diff(frame.index.asi8) == 10_000_000_000)
    assert frame["ut1_utc"].iloc[0] == pytest.approx(-0.018125)
    assert frame["quality"].iloc[2] == "DEGRADED-MANOEUVRE"


def test_merge_removes_overlap_and_keeps_earlier_file():
    a = eof.read_eof(FIXTURES / "eof_a.EOF")
    b = eof.read_eof(FIXTURES / "eof_b.EOF")
    b.loc[b.index[0], "x"] += 1.0  # distinguishable duplicate epoch
    merged = eof.merge([a, b])
    assert merged.index.is_unique
    assert len(merged) == 8
    assert merged.loc[b.index[0], "x"] == a.loc[b.index[0], "x"]


def test_find_gaps():
    frames = [eof.read_eof(FIXTURES / f"eof_{k}.EOF") for k in "abc"]
    gaps = eof.find_gaps(eof.merge(frames).index)
    assert len(gaps) == 1
    assert gaps["duration_s"].iloc[0] == pytest.approx(50.0)


def test_read_manoeuvres_packed_negative_fields():
    burns = esa_history.read_manoeuvres(FIXTURES / "s1x.man")
    assert len(burns) == 2
    first = burns.iloc[0]
    assert first["duration_s"] == pytest.approx(40.0)
    assert first["a1"] == pytest.approx(-0.43083620e-09)
    assert first["a2"] == pytest.approx(-0.47648873e-06)
    assert first["a3"] == pytest.approx(-0.16263033e-21)
    assert burns.iloc[1]["a2"] == pytest.approx(0.49307662e-06)


def test_read_manoeuvres_rejects_unpaired(tmp_path):
    lines = (FIXTURES / "s1x.man").read_text().splitlines()
    path = tmp_path / "bad.man"
    path.write_text("\n".join(lines[:2]))
    with pytest.raises(ValueError):
        esa_history.read_manoeuvres(path)


def test_read_outages_and_mass():
    out = esa_history.read_outages(FIXTURES / "s1x.out")
    assert list(out["kind"]) == ["GAP", "MAN"]
    mass = esa_history.read_mass_history(FIXTURES / "s1x.mhf")
    assert mass["mass_kg"].iloc[-1] == pytest.approx(2152.0)
    assert mass.index[-1] == pd.Timestamp("2014-07-27 14:36:16")


def test_spaceweather_parse_masks_missing_and_drops_sunspots():
    sw = spaceweather.parse(FIXTURES / "gfz_sample.txt")
    assert "sn" not in sw.columns
    assert sw.loc["2016-08-23", "f107_obs"] == pytest.approx(81.3)
    assert sw.loc["2016-08-23", "ap"] == pytest.approx(18)
    assert np.isnan(sw.loc["1932-01-06", "f107_obs"])


def test_poeorb_key_parsing_and_selection():
    keys = [
        "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20230101T081826_V20221211T225942_20221213T005942.EOF",
        "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20240301T000000_V20221211T225942_20221213T005942.EOF",
        "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20230102T081907_V20221212T225942_20221214T005942.EOF",
        "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20230103T081835_V20221213T225942_20221215T005942.EOF",
        "AUX_POEORB/not_an_orbit_file.txt",
    ]
    refs = [r for r in (poeorb.parse_key(k) for k in keys) if r is not None]
    assert len(refs) == 4
    assert refs[0].validity_start == datetime(2022, 12, 11, 22, 59, 42)
    selected = poeorb.select_window(refs, date(2022, 12, 12), date(2022, 12, 14))
    # Days 12 and 13 need the files starting on the 11th and the 12th; latest production wins.
    assert [r.validity_start.day for r in selected] == [11, 12]
    assert selected[0].created.year == 2024
