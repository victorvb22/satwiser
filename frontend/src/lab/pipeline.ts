/**
 * Degradation lab pipeline — TypeScript twin of `src/satwiser/lab.py`.
 *
 * Runs in a Web Worker. Every step mirrors the Python reference so that both give the
 * same numbers on the shared parity fixtures (`src/lab/__fixtures__/parity.json`).
 * Keep the two in step: any change here must be made in Python too, and vice versa.
 */

export const MU_EARTH = 3.986004418e14; // m^3 s^-2
export const R_EARTH = 6378137.0; // m
export const J2 = 1.08262668e-3;
export const STEP_S = 10.0; // POD grid
export const SAMPLES_PER_DAY = 86400 / STEP_S;
export const MIN_BINS = 6;
export const MIN_DENSE = 10;
export const PERIOD_S = 5924.0;

export interface Degradation {
  sigmaM: number;
  pointsPerDay: number;
  rho: number;
}

export interface Detector {
  windowRevs: number;
  threshold: number;
  normalisation: "window" | "diff";
  floorM: number;
}

/** States as six parallel arrays (inertial frame, m and m/s). */
export interface States {
  rx: Float64Array;
  ry: Float64Array;
  rz: Float64Array;
  vx: Float64Array;
  vy: Float64Array;
  vz: Float64Array;
}

export const STATE_KEYS = ["rx", "ry", "rz", "vx", "vy", "vz"] as const;

// ------------------------------------------------------------------------ randomness

/** Mulberry32: small seeded generator (uniform in [0, 1)). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draws (Box–Muller), row-major n x 6. */
export function normals(n: number, seed: number): Float64Array {
  const rand = mulberry32(seed);
  const out = new Float64Array(n * 6);
  for (let k = 0; k < out.length; k += 2) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[k] = r * Math.cos(2 * Math.PI * v);
    if (k + 1 < out.length) out[k + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

// ------------------------------------------------------------------------ degradation

/** AR(1) process with unit variance, applied column by column (row-major n x 6). */
export function ar1(z: Float64Array, n: number, rho: number): Float64Array {
  const out = new Float64Array(z.length);
  const scale = Math.sqrt(1 - rho * rho);
  for (let c = 0; c < 6; c++) {
    out[c] = z[c];
    for (let k = 1; k < n; k++) {
      out[k * 6 + c] = rho * out[(k - 1) * 6 + c] + scale * z[k * 6 + c];
    }
  }
  return out;
}

/** Indices of evenly spaced samples at `pointsPerDay` on a grid of `stepS` seconds. */
export function subsample(nSamples: number, pointsPerDay: number, phase: number,
                          stepS: number = STEP_S): Int32Array {
  const perDay = 86400 / stepS;
  const stride = Math.max(1, Math.round(perDay / pointsPerDay));
  const first = ((phase % stride) + stride) % stride;
  const count = first < nSamples ? Math.floor((nSamples - 1 - first) / stride) + 1 : 0;
  const idx = new Int32Array(count);
  for (let k = 0; k < count; k++) idx[k] = first + k * stride;
  return idx;
}

/** Correlated orbit-like errors: position sigma per axis, velocity n * sigma. */
export function degrade(states: States, idx: Int32Array, deg: Degradation,
                        z: Float64Array): States {
  const n = idx.length;
  const out: States = {
    rx: new Float64Array(n), ry: new Float64Array(n), rz: new Float64Array(n),
    vx: new Float64Array(n), vy: new Float64Array(n), vz: new Float64Array(n),
  };
  for (let k = 0; k < n; k++) {
    for (const key of STATE_KEYS) out[key][k] = states[key][idx[k]];
  }
  if (deg.sigmaM === 0) return out;
  let rMean = 0;
  for (let k = 0; k < n; k++) {
    rMean += Math.hypot(out.rx[k], out.ry[k], out.rz[k]);
  }
  rMean /= n;
  const meanMotion = Math.sqrt(MU_EARTH / rMean ** 3);
  const err = deg.rho === 0 ? z : ar1(z, n, deg.rho);
  for (let k = 0; k < n; k++) {
    out.rx[k] += deg.sigmaM * err[k * 6];
    out.ry[k] += deg.sigmaM * err[k * 6 + 1];
    out.rz[k] += deg.sigmaM * err[k * 6 + 2];
    out.vx[k] += deg.sigmaM * meanMotion * err[k * 6 + 3];
    out.vy[k] += deg.sigmaM * meanMotion * err[k * 6 + 4];
    out.vz[k] += deg.sigmaM * meanMotion * err[k * 6 + 5];
  }
  return out;
}

// ------------------------------------------------------------------ orbital elements

/**
 * Osculating semi-major axis minus the first-order J2 short-period term
 * (`a - 1.5 J2 R^2 / a sin^2 i cos 2u`), from inertial states.
 */
export function meanANosp(s: States): Float64Array {
  const n = s.rx.length;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const rx = s.rx[k], ry = s.ry[k], rz = s.rz[k];
    const vx = s.vx[k], vy = s.vy[k], vz = s.vz[k];
    const r = Math.hypot(rx, ry, rz);
    const v2 = vx * vx + vy * vy + vz * vz;
    const a = -MU_EARTH / (2 * (v2 / 2 - MU_EARTH / r));
    const hx = ry * vz - rz * vy;
    const hy = rz * vx - rx * vz;
    const hz = rx * vy - ry * vx;
    const h = Math.hypot(hx, hy, hz);
    const inc = Math.acos(Math.min(1, Math.max(-1, hz / h)));
    // Argument of latitude: angle from the ascending node to r, about h.
    const nx = -hy, ny = hx; // node vector (z component 0)
    const nNorm = Math.hypot(nx, ny);
    const cosU = (nx * rx + ny * ry) / nNorm;
    const cx = ny * rz, cy = -nx * rz, cz = nx * ry - ny * rx; // node x r
    const sinU = ((cx * hx + cy * hy + cz * hz) / h) / nNorm;
    const u = Math.atan2(sinU, cosU);
    const sinI = Math.sin(inc);
    out[k] = a - 1.5 * J2 * R_EARTH ** 2 / a * sinI * sinI * Math.cos(2 * u);
  }
  return out;
}

// ------------------------------------------------------------------------- binning

export function binRevolutions(pointsPerDay: number, periodS: number = PERIOD_S): number {
  const perRev = pointsPerDay * periodS / 86400;
  return perRev >= 1 ? 1 : Math.ceil(1 / perRev);
}

export function expectedPerBin(pointsPerDay: number, revsPerBin: number,
                               periodS: number = PERIOD_S): number {
  return pointsPerDay * revsPerBin * periodS / 86400;
}

/** Template-corrected mean per bin of revolutions (NaN for empty or truncated bins). */
export function binMeans(a: Float64Array, orbit: Int32Array, template: ArrayLike<number>,
                         revsPerBin: number, expected = 0): Float64Array {
  let oMin = Infinity, oMax = -Infinity;
  for (let k = 0; k < orbit.length; k++) {
    oMin = Math.min(oMin, orbit[k]);
    oMax = Math.max(oMax, orbit[k]);
  }
  const nBins = Math.floor((oMax - oMin) / revsPerBin) + 1;
  const sums = new Float64Array(nBins);
  const counts = new Float64Array(nBins);
  const repeat = template.length;
  for (let k = 0; k < a.length; k++) {
    const b = Math.floor((orbit[k] - oMin) / revsPerBin);
    sums[b] += a[k] - template[((orbit[k] % repeat) + repeat) % repeat];
    counts[b] += 1;
  }
  const means = new Float64Array(nBins).fill(NaN);
  for (let b = 0; b < nBins; b++) {
    let keep = counts[b] > 0;
    if (expected >= MIN_DENSE) keep = keep && counts[b] >= 0.9 * expected;
    if (keep) means[b] = sums[b] / counts[b];
  }
  return means;
}

// ------------------------------------------------------------------------ detection

function median(values: number[]): number {
  const s = [...values].sort((x, y) => x - y);
  const h = Math.floor(s.length / 2);
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

function mean(values: number[]): number {
  let t = 0;
  for (const v of values) t += v;
  return t / values.length;
}

/** Step statistic d and robust z-score; mirrors `lab.window_scores`. */
export function windowScores(means: Float64Array, window: number,
                             normalisation: Detector["normalisation"] = "window",
                             floorM = 0): { d: Float64Array; z: Float64Array } {
  const n = means.length;
  const d = new Float64Array(n).fill(NaN);
  const gap = new Float64Array(n).fill(NaN);
  const inv = new Float64Array(n).fill(NaN);
  const z = new Float64Array(n).fill(NaN);
  const minCount = Math.max(1, Math.floor((window + 1) / 2));
  for (let k = 0; k < n; k++) {
    const bIdx: number[] = [];
    const aIdx: number[] = [];
    for (let j = Math.max(0, k - window); j < k; j++) if (!Number.isNaN(means[j])) bIdx.push(j);
    for (let j = k + 1; j < Math.min(n, k + 1 + window); j++) {
      if (!Number.isNaN(means[j])) aIdx.push(j);
    }
    if (bIdx.length >= minCount && aIdx.length >= minCount) {
      d[k] = mean(aIdx.map((j) => means[j])) - mean(bIdx.map((j) => means[j]));
      gap[k] = mean(aIdx) - mean(bIdx);
      inv[k] = 1 / bIdx.length + 1 / aIdx.length;
    }
  }
  const okIdx: number[] = [];
  for (let k = 0; k < n; k++) if (!Number.isNaN(d[k])) okIdx.push(k);
  if (okIdx.length < MIN_BINS) return { d, z };

  if (normalisation === "window") {
    const vals = okIdx.map((k) => d[k]);
    const centre = median(vals);
    const scale = 1.4826 * median(vals.map((v) => Math.abs(v - centre)));
    if (scale > 0) for (const k of okIdx) z[k] = (d[k] - centre) / scale;
    return { d, z };
  }
  const valid: number[] = [];
  for (let k = 0; k < n; k++) if (!Number.isNaN(means[k])) valid.push(k);
  const steps: number[] = [];
  for (let j = 1; j < valid.length; j++) {
    if (valid[j] - valid[j - 1] === 1) steps.push(means[valid[j]] - means[valid[j - 1]]);
  }
  if (steps.length < 2) return { d, z };
  const drift = median(steps);
  const noise = 1.4826 * median(steps.map((s) => Math.abs(s - drift))) / Math.SQRT2;
  const varDrift = Math.PI / 2 * 2 * noise * noise / steps.length;
  for (const k of okIdx) {
    const scale = Math.max(Math.sqrt(noise * noise * inv[k] + gap[k] ** 2 * varDrift), floorM);
    if (scale > 0) z[k] = (d[k] - drift * gap[k]) / scale;
  }
  return { d, z };
}

/** Bin index of the |z| maximum of each run above the threshold. */
export function windowDetect(z: Float64Array, threshold: number, window: number): number[] {
  const above: number[] = [];
  for (let k = 0; k < z.length; k++) {
    if (!Number.isNaN(z[k]) && Math.abs(z[k]) > threshold) above.push(k);
  }
  const out: number[] = [];
  let run: number[] = [];
  const flush = () => {
    if (run.length) {
      out.push(run.reduce((best, k) => (Math.abs(z[k]) > Math.abs(z[best]) ? k : best), run[0]));
    }
    run = [];
  };
  for (const k of above) {
    if (run.length && k - run[run.length - 1] > window) flush();
    run.push(k);
  }
  flush();
  return out;
}

export function detectorWindow(det: Detector, revsPerBin: number): number {
  return Math.max(2, Math.round(det.windowRevs / revsPerBin));
}
