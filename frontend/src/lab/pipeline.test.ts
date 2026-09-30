/**
 * Parity with the Python reference (`src/satwiser/lab.py`): the fixture is written by
 * `scripts/make_parity_fixtures.py` from a real lab window, with fixed-seed normals.
 */
import fixture from "./__fixtures__/parity.json";
import {
  binMeans,
  binRevolutions,
  degrade,
  detectorWindow,
  expectedPerBin,
  meanANosp,
  mulberry32,
  normals,
  subsample,
  windowDetect,
  windowScores,
  type Detector,
  type States,
} from "./pipeline";

const states: States = {
  rx: Float64Array.from(fixture.states.rx),
  ry: Float64Array.from(fixture.states.ry),
  rz: Float64Array.from(fixture.states.rz),
  vx: Float64Array.from(fixture.states.vx),
  vy: Float64Array.from(fixture.states.vy),
  vz: Float64Array.from(fixture.states.vz),
};
const orbit = Int32Array.from(fixture.orbit);
const det: Detector = {
  windowRevs: fixture.detector.window_revs,
  threshold: fixture.detector.threshold,
  normalisation: fixture.detector.normalisation as Detector["normalisation"],
  floorM: fixture.detector.floor_m,
};

function expectClose(actual: ArrayLike<number>, expected: (number | null)[], tol: number) {
  expect(actual.length).toBe(expected.length);
  for (let k = 0; k < expected.length; k++) {
    const e = expected[k];
    if (e === null) expect(Number.isNaN(actual[k])).toBe(true);
    else expect(Math.abs(actual[k] - e)).toBeLessThanOrEqual(tol);
  }
}

describe.each(fixture.cases)("parity: $name", (c) => {
  const deg = { sigmaM: c.sigma_m, pointsPerDay: c.points_per_day, rho: c.rho };
  const idx = subsample(fixture.t_offset_s.length, deg.pointsPerDay, c.phase, fixture.step_s);

  it("selects the same samples", () => {
    expect(Array.from(idx)).toEqual(c.idx);
  });

  const noisy = degrade(states, idx, deg, Float64Array.from(c.normals));
  const a = meanANosp(noisy);
  const step = 2 * fixture.a_ref * c.dv_add / fixture.v_ref;
  for (let k = 0; k < a.length; k++) {
    if (fixture.t_offset_s[idx[k]] > fixture.event_s) a[k] += step;
  }

  it("computes the same J2-corrected semi-major axis", () => {
    expectClose(a, c.a_nosp, 1e-5);
  });

  const b = binRevolutions(deg.pointsPerDay);
  const sub = Int32Array.from(idx, (k) => orbit[k]);
  const means = binMeans(a, sub, fixture.template_a, b, expectedPerBin(deg.pointsPerDay, b));
  const w = detectorWindow(det, b);
  const { d, z } = windowScores(means, w, det.normalisation, det.floorM);

  it("bins and scores identically", () => {
    expect(b).toBe(c.revs_per_bin);
    expect(w).toBe(c.window);
    expectClose(means, c.means, 1e-5);
    expectClose(d, c.d, 1e-5);
    expectClose(z, c.z, 1e-6);
  });

  it("detects at the same bins", () => {
    expect(windowDetect(z, det.threshold, w)).toEqual(c.detections);
  });
});

describe("random draws", () => {
  it("are reproducible and standard normal", () => {
    expect(mulberry32(7)()).toBe(mulberry32(7)());
    const z = normals(20000, 1);
    let m = 0, v = 0;
    for (const x of z) m += x;
    m /= z.length;
    for (const x of z) v += (x - m) ** 2;
    v /= z.length;
    expect(Math.abs(m)).toBeLessThan(0.03);
    expect(Math.abs(v - 1)).toBeLessThan(0.05);
  });
});
