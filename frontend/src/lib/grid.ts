/**
 * Interpolation in the precomputed robustness grid (step 4). The lab never calls the API
 * while sliders move: detection probabilities and the 90 % threshold are read from the
 * grid, linearly interpolated in log(noise), log(Δv), log(points per day) and rho.
 */
import type { Robustness } from "../api/client";

/** Fractional index of `value` in an ascending or descending axis (clamped). */
export function axisPosition(axis: number[], value: number, log: boolean): number {
  const f = (v: number) => (log ? Math.log(v) : v);
  const ascending = axis[axis.length - 1] > axis[0];
  const a = ascending ? axis : [...axis].reverse();
  const v = f(value);
  let pos: number;
  if (v <= f(a[0])) pos = 0;
  else if (v >= f(a[a.length - 1])) pos = a.length - 1;
  else {
    let k = 0;
    while (f(a[k + 1]) < v) k++;
    pos = k + (v - f(a[k])) / (f(a[k + 1]) - f(a[k]));
  }
  return ascending ? pos : axis.length - 1 - pos;
}

export function nearestIndex(axis: number[], value: number, log: boolean): number {
  return Math.round(axisPosition(axis, value, log));
}

function blend(get: (i: number) => number | null, pos: number): number | null {
  const i0 = Math.floor(pos);
  const i1 = Math.min(i0 + 1, Math.ceil(pos));
  const w = pos - i0;
  const v0 = get(i0);
  const v1 = get(i1);
  if (v0 === null || v1 === null) return w < 0.5 ? v0 : v1;
  return v0 * (1 - w) + v1 * w;
}

/** P(detection) for every (sigma, Δv) cell at the current sampling and correlation. */
export function probabilityTable(grid: Robustness, pointsPerDay: number, rho: number
                                 ): number[][] {
  const pp = axisPosition(grid.axes.points_per_day, pointsPerDay, true);
  const pr = axisPosition(grid.axes.rho, rho, false);
  return grid.axes.sigma_m.map((_, si) =>
    grid.axes.dv_cm_s.map((__, di) =>
      blend((p) => blend((r) => grid.p_detect[si][di][p][r], pr), pp) ?? 0));
}

/** Smallest Δv detected at 90 % (cm/s) at the current setting; null if out of range. */
export function minDetectable(grid: Robustness, sigma: number, pointsPerDay: number,
                              rho: number): number | null {
  const ps = axisPosition(grid.axes.sigma_m, sigma, true);
  const pp = axisPosition(grid.axes.points_per_day, pointsPerDay, true);
  const pr = axisPosition(grid.axes.rho, rho, false);
  const logOrNull = (v: number | null) => (v === null ? null : Math.log(v));
  const value = blend((s) => blend((p) => blend((r) =>
    logOrNull(grid.min_dv_90_cm_s[s][p][r]), pr), pp), ps);
  return value === null ? null : Math.exp(value);
}
