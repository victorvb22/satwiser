"""Classification of detected events (applied to detections only).

Classes:

- ``station_keeping``: matched to an ESA manoeuvre of type ``station_keeping``;
- ``orbit_change``: matched to an ESA manoeuvre of another type (inclination, multi-burn
  sequence, lowering);
- ``unexplained``: no ESA manoeuvre within the matching tolerance. The ESA record has no
  anomaly class, so this class is defined operationally as "detection without a logged
  manoeuvre" (possible unlogged event, orbit-determination artefact or false alarm).

Two classifiers are compared on a strict temporal split: a documented rule on the
estimated jumps, and a gradient-boosted tree model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = ("station_keeping", "orbit_change", "unexplained")
FEATURES = ("da", "de", "di", "abs_da", "abs_di", "dv_est", "stat_abs", "alarm_lag",
            "on_i_channel", "f107_81d", "ap", "days_since_prev")


def label_detections(d: pd.DataFrame, m: pd.DataFrame) -> pd.Series:
    """Class of each detection from the evaluation tables of :func:`evaluate`."""
    types = m["type"].reindex(d["manoeuvre"].to_numpy()).to_numpy()
    out = np.where(d["manoeuvre"].to_numpy() < 0, "unexplained",
                   np.where(types == "station_keeping", "station_keeping", "orbit_change"))
    return pd.Series(out, index=d.index)


def features(d: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Feature table for detections carrying CUSUM jump estimates (da, de, di...)."""
    x = pd.DataFrame(index=d.index)
    x["da"] = d["da"]
    x["de"] = d["de"]
    x["di"] = d["di"]
    x["abs_da"] = d["da"].abs()
    x["abs_di"] = d["di"].abs()
    x["dv_est"] = d["dv_est"]
    x["stat_abs"] = d["z"].abs()
    x["alarm_lag"] = d["alarm_orbit"] - d["orbit"]
    x["on_i_channel"] = (d["channel"] == "i").astype(float)
    sw = grid[["f107_81d", "ap"]].reindex(d["orbit"].to_numpy())
    x["f107_81d"] = sw["f107_81d"].to_numpy()
    x["ap"] = sw["ap"].to_numpy()
    t = pd.to_datetime(d["time"])
    order = np.argsort(t.to_numpy())
    gaps = np.diff(t.to_numpy()[order]).astype("timedelta64[s]").astype(float) / 86400
    since = np.full(len(d), np.nan)
    since[order[1:]] = gaps
    x["days_since_prev"] = since
    return x[list(FEATURES)]


def rule(x: pd.DataFrame, di_threshold_mdeg: float, da_threshold_m: float) -> np.ndarray:
    """Documented rule baseline.

    - a significant inclination jump (|Δi| > ``di_threshold_mdeg``) or a negative Δa
      means an orbit change;
    - a positive Δa above ``da_threshold_m`` is station keeping;
    - anything smaller on both channels is unexplained.
    """
    orbit_change = (x["abs_di"] > di_threshold_mdeg) | (x["da"] < -da_threshold_m)
    sk = ~orbit_change & (x["da"] > da_threshold_m)
    return np.where(orbit_change, "orbit_change", np.where(sk, "station_keeping", "unexplained"))


def train_model(x: pd.DataFrame, y: pd.Series, seed: int = 0) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                           class_weight="balanced", random_state=seed)
    return model.fit(x, y)


def report(y_true, y_pred) -> dict:
    labels = list(CLASSES)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "confusion": pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                                  index=[f"true {c}" for c in labels],
                                  columns=[f"pred {c}" for c in labels]),
        "per_class": pd.DataFrame(classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0)).T.loc[labels],
    }
