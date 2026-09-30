"""Ground-truth manoeuvres from the ESA manoeuvre history.

The ``.man`` file lists individual burns. Burns less than ``MAX_GAP_H`` hours apart are
grouped into one manoeuvre, the unit used for evaluation. The gaps between consecutive
Sentinel-1A burns are bimodal: within an operation they are below 5 h (typically half a
revolution or a few revolutions), between operations above 8 h, with none in 5-8 h. The
6 h threshold sits in that empty band (``reports/step2/baseline.md``).

The accelerations are in km/s^2 with components (radial, along-track, cross-track);
this was checked against the orbit in the step 1 reconnaissance
(``reports/step1/recon.md``).

The file carries no manoeuvre type. A type is derived from the burn geometry with a
documented rule (:func:`manoeuvre_type`) rather than invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from satwiser.orbit.elements import MU_EARTH

MAX_GAP_H = 6.0
A_REF = 7_071_000.0  # m, Sentinel-1 reference semi-major axis
V_REF = float(np.sqrt(MU_EARTH / A_REF))

TYPES = ("station_keeping", "inclination", "sequence", "lowering")


def cluster_burns(burns: pd.DataFrame, max_gap_h: float = MAX_GAP_H) -> np.ndarray:
    """Label burns separated by less than ``max_gap_h`` hours as one manoeuvre."""
    gaps = burns["start"].diff() > pd.Timedelta(hours=max_gap_h)
    return np.cumsum(gaps.to_numpy())


def manoeuvre_type(dv_t: np.ndarray, dv_n: np.ndarray) -> str:
    """Classify a manoeuvre from the signed along-track and cross-track burn Δv (m/s).

    - ``inclination``: cross-track Δv dominates (out-of-plane burn);
    - ``sequence``: along-track burns of both signs (multi-burn in-plane sequence whose
      net semi-major axis change is small, e.g. phasing or avoidance);
    - ``station_keeping``: along-track burns all positive (drag make-up);
    - ``lowering``: along-track burns all negative.
    """
    if np.abs(dv_n).sum() > np.abs(dv_t).sum():
        return "inclination"
    significant = dv_t[np.abs(dv_t) > 0.1 * np.abs(dv_t).max()]
    if (significant > 0).any() and (significant < 0).any():
        return "sequence"
    return "station_keeping" if dv_t.sum() > 0 else "lowering"


def manoeuvres(burns: pd.DataFrame) -> pd.DataFrame:
    """One row per manoeuvre with its timing, burn count, Δv and expected Δa.

    ``dv_t`` is the signed along-track Δv, ``dv_abs`` the total |Δv| spent, and
    ``da_pred_m`` the semi-major axis change predicted by the Gauss equation
    ``Δa = 2 a Δv_T / v`` for a near-circular orbit.
    """
    b = burns.copy()
    b["dv_t"] = b["a2"] * 1e3 * b["duration_s"]
    b["dv_n"] = b["a3"] * 1e3 * b["duration_s"]
    b["dv_r"] = b["a1"] * 1e3 * b["duration_s"]
    b["cluster"] = cluster_burns(b)
    rows = []
    for _, g in b.groupby("cluster"):
        dv_t = g["dv_t"].to_numpy()
        dv_n = g["dv_n"].to_numpy()
        rows.append({
            "start": g["start"].min(),
            "stop": g["stop"].max(),
            "n_burns": len(g),
            "dv_t": dv_t.sum(),
            "dv_n_abs": np.abs(dv_n).sum(),
            "dv_abs": np.sqrt(g["dv_t"] ** 2 + g["dv_n"] ** 2 + g["dv_r"] ** 2).sum(),
            "da_pred_m": 2 * A_REF * dv_t.sum() / V_REF,
            "type": manoeuvre_type(dv_t, dv_n),
        })
    return pd.DataFrame(rows)


def dv_from_da(da_m: np.ndarray | float, a: float = A_REF) -> np.ndarray | float:
    """Along-track Δv (m/s) equivalent to a semi-major axis jump: ``Δv = Δa v / (2a)``."""
    return da_m * np.sqrt(MU_EARTH / a) / (2 * a)
