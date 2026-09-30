"""State vectors to classical orbital elements.

POEORB vectors are Earth-fixed. They are rotated to a quasi-inertial frame by the
Greenwich mean sidereal angle computed from UT1, and the velocity is corrected for the
Earth's rotation (``v_i = R (v_e + omega x r_e)``). Precession, nutation and polar motion
are neglected: this frame is close to true-of-date, which only affects the inclination
at the millidegree level and leaves the semi-major axis and eccentricity unchanged (they
depend on ``|r|``, ``|v_i|`` and ``r . v_i`` only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MU_EARTH = 3.986004418e14  # m^3 s^-2 (EGM2008 / WGS84)
OMEGA_EARTH = 7.292115e-5  # rad s^-1
R_EARTH = 6378137.0  # m, equatorial radius
J2 = 1.08262668e-3

_J2000 = np.datetime64("2000-01-01T12:00:00", "us")


def gmst(ut1: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal angle (rad) for UT1 epochs (IAU 1982 expression)."""
    d = (ut1.astype("datetime64[us]") - _J2000).astype(np.int64) * 1e-6 / 86400.0
    t = d / 36525.0
    seconds = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t
               + 0.093104 * t**2 - 6.2e-6 * t**3)
    return np.mod(np.deg2rad(seconds / 240.0), 2.0 * np.pi)


def ecef_to_inertial(r_e: np.ndarray, v_e: np.ndarray, theta: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate Earth-fixed position/velocity (N x 3) by angle ``theta`` about +Z."""
    c, s = np.cos(theta), np.sin(theta)
    v_rel = v_e + np.cross(np.array([0.0, 0.0, OMEGA_EARTH]), r_e)

    def rot(u: np.ndarray) -> np.ndarray:
        return np.column_stack((c * u[:, 0] - s * u[:, 1], s * u[:, 0] + c * u[:, 1], u[:, 2]))

    return rot(r_e), rot(v_rel)


def keplerian(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> dict[str, np.ndarray]:
    """Osculating elements from inertial position (m) and velocity (m/s), both N x 3.

    Returns ``a`` (m), ``e``, ``i``, ``raan``, ``argp``, ``nu`` and ``u`` (argument of
    latitude), angles in radians. ``argp`` and ``nu`` are ill-conditioned for the
    near-circular orbits considered here; ``u`` is always well defined.
    """
    rn = np.linalg.norm(r, axis=1)
    vn2 = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)
    energy = vn2 / 2.0 - mu / rn
    a = -mu / (2.0 * energy)

    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    e_vec = ((vn2 - mu / rn)[:, None] * r - rv[:, None] * v) / mu
    e = np.linalg.norm(e_vec, axis=1)
    inc = np.arccos(np.clip(h[:, 2] / hn, -1.0, 1.0))

    node = np.column_stack((-h[:, 1], h[:, 0], np.zeros(len(h))))
    nn = np.linalg.norm(node, axis=1)
    raan = np.mod(np.arctan2(node[:, 1], node[:, 0]), 2.0 * np.pi)

    def angle(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Signed angle from p to q about h.
        cos = np.einsum("ij,ij->i", p, q)
        sin = np.einsum("ij,ij->i", np.cross(p, q), h) / hn
        return np.mod(np.arctan2(sin, cos), 2.0 * np.pi)

    node_u = node / nn[:, None]
    argp = angle(node_u, e_vec)
    nu = angle(e_vec, r)
    u = angle(node_u, r)
    return {"a": a, "e": e, "i": inc, "raan": raan, "argp": argp, "nu": nu, "u": u}


def j2_short_period_a(a: np.ndarray, i: np.ndarray, u: np.ndarray) -> np.ndarray:
    """First-order J2 short-period term of the semi-major axis for a near-circular orbit.

    ``a_osc = a_mean + 3 J2 R^2 / (2 a) * sin(i)^2 * cos(2u)`` (Kozai 1962, e -> 0).
    For Sentinel-1 the amplitude is about 9.15 km. Subtracting it before averaging makes
    the per-revolution mean insensitive to where the revolution boundaries fall on the
    10 s sampling grid (otherwise a ~15 m revolution-to-revolution jitter).
    """
    return 1.5 * J2 * R_EARTH**2 / a * np.sin(i) ** 2 * np.cos(2.0 * u)


def elements_from_eof(frame: pd.DataFrame) -> pd.DataFrame:
    """Osculating elements for a frame produced by :func:`satwiser.io.eof.read_eof`.

    ``a_nosp`` is the osculating semi-major axis minus the J2 short-period term.
    """
    utc = frame.index.values.astype("datetime64[us]")
    ut1 = utc + (frame["ut1_utc"].to_numpy() * 1e6).astype("timedelta64[us]")
    r_e = frame[["x", "y", "z"]].to_numpy()
    v_e = frame[["vx", "vy", "vz"]].to_numpy()
    r, v = ecef_to_inertial(r_e, v_e, gmst(ut1))
    elements = pd.DataFrame(keplerian(r, v), index=frame.index)
    elements["a_nosp"] = elements["a"] - j2_short_period_a(
        elements["a"].to_numpy(), elements["i"].to_numpy(), elements["u"].to_numpy())
    elements["orbit"] = frame["orbit"].to_numpy()
    return elements
