"""Download and parse daily solar and geomagnetic indices from GFZ Potsdam.

File: ``Kp_ap_Ap_SN_F107_since_1932.txt`` (https://kp.gfz.de). Kp, ap, Ap and F10.7 are
distributed under CC BY 4.0. The sunspot number column is CC BY-NC 4.0 and is dropped
on parsing so that it never ends up in derived products.

Citation: Matzka, J., Stolle, C., Yamazaki, Y., Bronkalla, O. and Morschhauser, A., 2021.
The geomagnetic Kp index and derived indices of geomagnetic activity. Space Weather,
https://doi.org/10.1029/2020SW002641. F10.7 is produced by DRAO / NRCan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
FILENAME = "Kp_ap_Ap_SN_F107_since_1932.txt"

_COLUMNS = (
    ["year", "month", "day", "days", "days_m", "bsr", "db"]
    + [f"kp{i}" for i in range(1, 9)]
    + [f"ap{i}" for i in range(1, 9)]
    + ["ap_daily", "sn", "f107_obs", "f107_adj", "definitive"]
)


def download(dest: Path, overwrite: bool = False) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / FILENAME
    if overwrite or not path.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def parse(path: Path) -> pd.DataFrame:
    """Return a daily table indexed by UTC date with ``f107_obs``, ``f107_adj``, ``ap``.

    Missing values (encoded -1 upstream) become NaN.
    """
    raw = pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=_COLUMNS)
    out = pd.DataFrame(
        {
            "f107_obs": raw["f107_obs"].astype(float),
            "f107_adj": raw["f107_adj"].astype(float),
            "ap": raw["ap_daily"].astype(float),
            "kp_definitive": raw["definitive"] >= 1,
        }
    )
    out.index = pd.DatetimeIndex(pd.to_datetime(raw[["year", "month", "day"]]), name="date")
    for column in ("f107_obs", "f107_adj", "ap"):
        out.loc[out[column] < 0, column] = np.nan
    return out
