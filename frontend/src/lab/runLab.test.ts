/** Worker logic on a real window slice (parity fixture): verdicts and bookkeeping. */
import type { LabWindow } from "../api/client";
import fixture from "./__fixtures__/parity.json";
import { prepare, runLab } from "./runLab";

const start = "2024-01-01T00:00:00";
const eventIso = new Date(Date.parse(`${start}Z`) + fixture.event_s * 1000).toISOString().slice(0, 19);
const window: LabWindow = {
  event_id: "test", satellite: "S1A", start, step_s: fixture.step_s, event_time: eventIso,
  t_offset_s: fixture.t_offset_s, orbit: fixture.orbit,
  states: fixture.states as LabWindow["states"], template_a: fixture.template_a,
  detector: fixture.detector as LabWindow["detector"],
  esa_manoeuvres: [{ start: eventIso, dv_t_mm_s: 5, type: "station_keeping" }],
};
const prepared = prepare(window, null);

describe("runLab", () => {
  it("takes the real Δv from the ESA manoeuvre at the event", () => {
    expect(prepared.realDvMmS).toBe(5);
    expect(prepared.otherSpansS).toEqual([]);
    expect(prepared.eventSpanS[0]).toBeCloseTo(fixture.event_s, 3);
  });

  it("detects the real manoeuvre with precise orbits", () => {
    const r = runLab(prepared, { sigmaM: 0.03, pointsPerDay: 1440, rho: 0, dvMmS: 5, seed: 1 });
    expect(r.verdict).toBe("detected");
    expect(r.detections.some((d) => d.nearEvent)).toBe(true);
    expect(r.stepM).toBeCloseTo(2 * 7_071_000 * 0.005 / Math.sqrt(3.986004418e14 / 7_071_000), 6);
    expect(r.means.length).toBe(r.tDays.length);
  });

  it("loses a small manoeuvre in TLE-like noise", () => {
    const r = runLab(prepared, { sigmaM: 1000, pointsPerDay: 2, rho: 0.9, dvMmS: 5, seed: 1 });
    expect(["missed", "insufficient"]).toContain(r.verdict);
    expect(r.revsPerBin).toBeGreaterThan(1);
  });

  it("removing the manoeuvre leaves nothing to detect near the event", () => {
    const r = runLab(prepared, { sigmaM: 0.03, pointsPerDay: 1440, rho: 0, dvMmS: 0, seed: 1 });
    expect(r.detections.some((d) => d.nearEvent)).toBe(false);
  });

  it("scores a detection against the ESA span, not the detector's change point", () => {
    // Event time two hours (more than one revolution) before the real manoeuvre: the
    // detection must still count, because the ESA record places the jump.
    const early = new Date(Date.parse(`${eventIso}Z`) - 2 * 3600 * 1000).toISOString().slice(0, 19);
    const shifted = prepare({ ...window, event_time: early }, null);
    expect(shifted.eventSpanS[0]).toBeCloseTo(fixture.event_s, 3);
    const r = runLab(shifted, { sigmaM: 0.03, pointsPerDay: 1440, rho: 0, dvMmS: 5, seed: 1 });
    expect(r.verdict).toBe("detected");
    expect(r.eventDays * 86400).toBeCloseTo(fixture.event_s, 3);
  });

  it("is reproducible for a given seed", () => {
    const p = { sigmaM: 10, pointsPerDay: 144, rho: 0.5, dvMmS: 5, seed: 7 };
    expect(runLab(prepared, p).means).toEqual(runLab(prepared, p).means);
  });
});
