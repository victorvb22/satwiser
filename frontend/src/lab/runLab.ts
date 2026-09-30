/**
 * One lab evaluation on a real ten-day window: degrade, average per bin, detect, and
 * judge the result against the event and the ESA manoeuvres in the window.
 */
import type { LabWindow } from "../api/client";
import {
  binMeans,
  binRevolutions,
  degrade,
  detectorWindow,
  expectedPerBin,
  meanANosp,
  MU_EARTH,
  normals,
  STATE_KEYS,
  subsample,
  windowDetect,
  windowScores,
  type Degradation,
  type Detector,
  type States,
} from "./pipeline";

export const A_REF = 7_071_000.0;
export const V_REF = Math.sqrt(MU_EARTH / A_REF);

export interface PreparedWindow {
  states: States;
  orbit: Int32Array;
  tOffsetS: Float64Array;
  stepS: number;
  eventS: number;
  template: Float64Array;
  detector: Detector;
  manoeuvresS: number[];
  realDvMmS: number;
}

export interface LabParams extends Degradation {
  /** Along-track Δv of the event's manoeuvre shown in the lab (mm/s). */
  dvMmS: number;
  seed: number;
}

export type Verdict = "detected" | "missed" | "quiet" | "false_alarm" | "insufficient";

export interface LabResult {
  tDays: number[];
  means: (number | null)[];
  clean: (number | null)[];
  z: (number | null)[];
  detections: { bin: number; tDays: number; nearEvent: boolean; nearOther: boolean }[];
  eventDays: number;
  verdict: Verdict;
  zMaxNearEvent: number | null;
  noisePerBinM: number | null;
  stepM: number;
  revsPerBin: number;
  window: number;
  nSamples: number;
}

export function prepare(win: LabWindow, realDvMmS: number | null): PreparedWindow {
  const start = Date.parse(win.start.endsWith("Z") ? win.start : `${win.start}Z`);
  const toS = (iso: string) => (Date.parse(iso.endsWith("Z") ? iso : `${iso}Z`) - start) / 1000;
  const states = Object.fromEntries(
    STATE_KEYS.map((k) => [k, Float64Array.from(win.states[k])])) as unknown as States;
  const eventS = toS(win.event_time);
  const nearest = win.esa_manoeuvres
    .map((m) => ({ s: toS(m.start), dv: m.dv_t_mm_s }))
    .sort((p, q) => Math.abs(p.s - eventS) - Math.abs(q.s - eventS))[0];
  const own = nearest && Math.abs(nearest.s - eventS) < 3 * 3600 ? nearest : null;
  return {
    states,
    orbit: Int32Array.from(win.orbit),
    tOffsetS: Float64Array.from(win.t_offset_s),
    stepS: win.step_s,
    eventS,
    template: Float64Array.from(win.template_a),
    detector: {
      windowRevs: win.detector.window_revs,
      threshold: win.detector.threshold,
      normalisation: win.detector.normalisation,
      floorM: win.detector.floor_m,
    },
    manoeuvresS: win.esa_manoeuvres.map((m) => toS(m.start)).filter((s) => !own || s !== own.s),
    realDvMmS: realDvMmS ?? own?.dv ?? 0,
  };
}

const mmToDa = (mm: number) => 2 * A_REF * (mm * 1e-3) / V_REF;

function seriesFor(p: PreparedWindow, idx: Int32Array, a: Float64Array, revsPerBin: number,
                   expected: number) {
  const orbit = Int32Array.from(idx, (k) => p.orbit[k]);
  return binMeans(a, orbit, p.template, revsPerBin, expected);
}

export function runLab(p: PreparedWindow, params: LabParams): LabResult {
  const deg: Degradation = { sigmaM: params.sigmaM, pointsPerDay: params.pointsPerDay,
                             rho: params.rho };
  const idx = subsample(p.tOffsetS.length, deg.pointsPerDay, params.seed, p.stepS);
  const stepM = mmToDa(params.dvMmS - p.realDvMmS);
  const withStep = (a: Float64Array) => {
    for (let k = 0; k < a.length; k++) if (p.tOffsetS[idx[k]] > p.eventS) a[k] += stepM;
    return a;
  };
  const noisy = withStep(meanANosp(degrade(p.states, idx, deg, normals(idx.length, params.seed))));
  const clean = withStep(meanANosp(degrade(p.states, idx, { ...deg, sigmaM: 0 },
                                           new Float64Array(0))));
  const revsPerBin = binRevolutions(deg.pointsPerDay);
  const expected = expectedPerBin(deg.pointsPerDay, revsPerBin);
  const means = seriesFor(p, idx, noisy, revsPerBin, expected);
  const cleanMeans = seriesFor(p, idx, clean, revsPerBin, expected);
  const window = detectorWindow(p.detector, revsPerBin);
  const { z } = windowScores(means, window, p.detector.normalisation, p.detector.floorM);

  // Time of each bin: mean epoch of its samples.
  const oMin = Math.min(...Array.from(idx, (k) => p.orbit[k]));
  const tSum = new Float64Array(means.length);
  const tCount = new Float64Array(means.length);
  for (const k of idx) {
    const b = Math.floor((p.orbit[k] - oMin) / revsPerBin);
    tSum[b] += p.tOffsetS[k];
    tCount[b] += 1;
  }
  const tDays = Array.from(tSum, (s, b) => (tCount[b] ? s / tCount[b] / 86400 : NaN));
  const binOfTime = (s: number) => {
    let best = 0;
    for (let b = 0; b < tDays.length; b++) {
      if (Math.abs(tDays[b] * 86400 - s) < Math.abs(tDays[best] * 86400 - s)) best = b;
    }
    return best;
  };
  const eventBin = binOfTime(p.eventS);
  const otherBins = p.manoeuvresS.map(binOfTime);
  const detections = windowDetect(z, p.detector.threshold, window).map((bin) => ({
    bin,
    tDays: tDays[bin],
    nearEvent: Math.abs(bin - eventBin) <= 1,
    nearOther: otherBins.some((o) => Math.abs(o - bin) <= 1),
  }));

  let zMax: number | null = null;
  for (let b = Math.max(0, eventBin - 1); b <= Math.min(z.length - 1, eventBin + 1); b++) {
    if (!Number.isNaN(z[b]) && (zMax === null || Math.abs(z[b]) > Math.abs(zMax))) zMax = z[b];
  }
  const allNaN = Array.from(z).every(Number.isNaN);
  const hasManoeuvre = Math.abs(params.dvMmS) > 0;
  const spurious = detections.some((d) => !d.nearEvent && !d.nearOther);
  let verdict: Verdict;
  if (allNaN) verdict = "insufficient";
  else if (detections.some((d) => d.nearEvent)) verdict = hasManoeuvre ? "detected" : "false_alarm";
  else if (hasManoeuvre) verdict = "missed";
  else verdict = spurious ? "false_alarm" : "quiet";

  const valid: number[] = [];
  for (let b = 0; b < means.length; b++) if (!Number.isNaN(means[b])) valid.push(b);
  const steps: number[] = [];
  for (let j = 1; j < valid.length; j++) {
    if (valid[j] - valid[j - 1] === 1) steps.push(means[valid[j]] - means[valid[j - 1]]);
  }
  let noise: number | null = null;
  if (steps.length >= 2) {
    const sorted = [...steps].sort((x, y) => x - y);
    const med = sorted[Math.floor(sorted.length / 2)];
    const dev = steps.map((s) => Math.abs(s - med)).sort((x, y) => x - y);
    noise = 1.4826 * dev[Math.floor(dev.length / 2)] / Math.SQRT2;
  }
  const toNullable = (arr: Float64Array) => Array.from(arr, (v) => (Number.isNaN(v) ? null : v));
  return {
    tDays,
    means: toNullable(means),
    clean: toNullable(cleanMeans),
    z: toNullable(z),
    detections,
    eventDays: p.eventS / 86400,
    verdict,
    zMaxNearEvent: zMax,
    noisePerBinM: noise,
    stepM: mmToDa(params.dvMmS),
    revsPerBin,
    window,
    nSamples: idx.length,
  };
}
