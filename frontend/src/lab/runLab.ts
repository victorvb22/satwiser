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
  median,
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
  /** Span of the studied manoeuvre (seconds from the window start): first burn start to
   * last burn end from the ESA record, or the event time for an unlabelled detection. */
  eventSpanS: Span;
  template: Float64Array;
  detector: Detector;
  /** Spans of the other ESA manoeuvres of the window. */
  otherSpansS: Span[];
  realDvMmS: number;
}

type Span = [number, number];

export interface LabParams extends Degradation {
  /** Along-track Δv of the event's manoeuvre shown in the lab (mm/s); 0 removes it. */
  dvMmS: number;
  /** Seed of the noise draw. The sampling phase is fixed so that σ = 0 is reproducible. */
  seed: number;
}

/** Central lab configuration (served with the robustness grid), overriding the copy
 * embedded in each window so that one source of truth drives every window. */
export interface LabConfig {
  detector?: Detector;
  template?: ArrayLike<number>;
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

export function prepare(win: LabWindow, realDvMmS: number | null,
                        config: LabConfig = {}): PreparedWindow {
  const start = Date.parse(win.start.endsWith("Z") ? win.start : `${win.start}Z`);
  const toS = (iso: string) => (Date.parse(iso.endsWith("Z") ? iso : `${iso}Z`) - start) / 1000;
  const states = Object.fromEntries(
    STATE_KEYS.map((k) => [k, Float64Array.from(win.states[k])])) as unknown as States;
  const eventS = toS(win.event_time);
  const spans = win.esa_manoeuvres.map((m) => ({
    span: [toS(m.start), toS(m.stop ?? m.start)] as Span, dv: m.dv_t_mm_s }));
  const nearest = [...spans]
    .sort((p, q) => Math.abs(p.span[0] - eventS) - Math.abs(q.span[0] - eventS))[0];
  // The event time is the detector's change point; the ESA record gives the truth.
  const own = nearest && Math.abs(nearest.span[0] - eventS) < 3 * 3600 ? nearest : null;
  return {
    states,
    orbit: Int32Array.from(win.orbit),
    tOffsetS: Float64Array.from(win.t_offset_s),
    stepS: win.step_s,
    eventSpanS: own ? own.span : [eventS, eventS],
    template: Float64Array.from(config.template ?? win.template_a),
    detector: config.detector ?? {
      windowRevs: win.detector.window_revs,
      threshold: win.detector.threshold,
      normalisation: win.detector.normalisation,
      floorM: win.detector.floor_m,
    },
    otherSpansS: spans.filter((m) => m !== own).map((m) => m.span),
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
  const idx = subsample(p.tOffsetS.length, deg.pointsPerDay, 0, p.stepS);
  const stepM = mmToDa(params.dvMmS - p.realDvMmS);
  // A changed Δv is applied where the real manoeuvre starts, on top of its own jump.
  const startS = p.eventSpanS[0];
  const withStep = (a: Float64Array) => {
    for (let k = 0; k < a.length; k++) if (p.tOffsetS[idx[k]] > startS) a[k] += stepM;
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
  const oMin = p.orbit[idx[0]]; // orbit numbers increase with time
  const tSum = new Float64Array(means.length);
  const tCount = new Float64Array(means.length);
  for (const k of idx) {
    const b = Math.floor((p.orbit[k] - oMin) / revsPerBin);
    tSum[b] += p.tOffsetS[k];
    tCount[b] += 1;
  }
  const tDays = Array.from(tSum, (s, b) => (tCount[b] ? s / tCount[b] / 86400 : NaN));
  // Bin of the revolution flown at time s, counted from the first sampled orbit, and a
  // detection matches a manoeuvre within one bin of its span: the step 4 replay rule.
  const binAt = (s: number) => {
    let lo = 0, hi = p.tOffsetS.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (p.tOffsetS[mid] < s) lo = mid + 1; else hi = mid;
    }
    return Math.floor((p.orbit[lo] - oMin) / revsPerBin);
  };
  const toBins = ([s0, s1]: Span): Span => [binAt(s0), binAt(s1)];
  const near = (bin: number, [k0, k1]: Span) => k0 - 1 <= bin && bin <= k1 + 1;
  const eventBins = toBins(p.eventSpanS);
  const otherBins = p.otherSpansS.map(toBins);
  const detections = windowDetect(z, p.detector.threshold, window).map((bin) => ({
    bin,
    tDays: tDays[bin],
    nearEvent: near(bin, eventBins),
    nearOther: otherBins.some((o) => near(bin, o)),
  }));

  let zMax: number | null = null;
  for (let b = Math.max(0, eventBins[0] - 1); b <= Math.min(z.length - 1, eventBins[1] + 1); b++) {
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
    const med = median(steps);
    noise = 1.4826 * median(steps.map((s) => Math.abs(s - med))) / Math.SQRT2;
  }
  const toNullable = (arr: Float64Array) => Array.from(arr, (v) => (Number.isNaN(v) ? null : v));
  return {
    tDays,
    means: toNullable(means),
    clean: toNullable(cleanMeans),
    z: toNullable(z),
    detections,
    eventDays: startS / 86400,
    verdict,
    zMaxNearEvent: zMax,
    noisePerBinM: noise,
    stepM: mmToDa(params.dvMmS),
    revsPerBin,
    window,
    nSamples: idx.length,
  };
}
