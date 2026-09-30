/** Render test of the two main screens against a mocked API and an in-thread worker. */
import { QueryClient } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { createMemoryRouter } from "react-router-dom";

import App, { routes } from "../App";
import fixture from "../lab/__fixtures__/parity.json";
import { prepare, runLab } from "../lab/runLab";
import type { WorkerRequest } from "../lab/worker";

const counts = { esa_manoeuvres: 3, detected: 2, missed: 1, false_alarms: 1 };
const start = "2025-03-01T00:00:00";
const eventIso = new Date(Date.parse(`${start}Z`) + fixture.event_s * 1000).toISOString().slice(0, 19);
const t0 = Date.UTC(2025, 2, 1);

const detected = {
  id: "S1A-D100", kind: "detected", time: "2025-03-01T03:00:00Z", orbit: 100,
  class: "station_keeping", dv_est_mm_s: 5.2, dv_esa_mm_s: 5.0, da_m: 9.8, lab_available: true,
};

const falseAlarm = {
  id: "S1A-F200", kind: "false_alarm", time: "2025-03-01T03:00:00Z", orbit: 100,
  class: "unexplained", dv_est_mm_s: 5.1, dv_esa_mm_s: null, da_m: 9.6, lab_available: true,
};

const labWindow = (id: string, manoeuvres: unknown[]) => ({
  event_id: id, satellite: "S1A", start, step_s: fixture.step_s, event_time: eventIso,
  t_offset_s: fixture.t_offset_s, orbit: fixture.orbit, states: fixture.states,
  template_a: fixture.template_a, detector: fixture.detector, esa_manoeuvres: manoeuvres,
});

const payloads: Record<string, unknown> = {
  "/api/satellites": [{
    id: "S1A", name: "Sentinel-1A", first: "2014-04-06T00:00:00", last: "2026-06-30T00:00:00",
    operational_start: "2014-08-01", split: "2020-01-01", labels_end: "2026-06-13T00:00:00",
    years: [2024, 2025, 2026], default_lab_event: "S1A-D100",
    summary: { all: counts, by_split: { test: counts }, by_year: { "2025": counts } },
  }],
  "/api/satellites/S1A/series?year=2025": {
    satellite: "S1A", year: 2025, orbit: [98, 99, 100, 101, 102],
    t_ms: [0, 1, 2, 3, 4].map((k) => t0 + k * 5.9e6),
    a_m: [7071000, 7070999.8, 7071010, 7071009.8, 7071009.6], f107: [150, 150, 151, 151, 152],
  },
  "/api/satellites/S1A/overview": {
    satellite: "S1A", t_ms: [t0, t0 + 8.64e7], a_m: [7071000, 7071001], f107: [150, 151],
  },
  "/api/satellites/S1A/events?year=2025": [
    detected,
    { id: "S1A-M101", kind: "missed", time: "2025-03-01T05:00:00Z", orbit: 101,
      class: "orbit_change", dv_est_mm_s: null, dv_esa_mm_s: 1.0, da_m: null,
      lab_available: false },
  ],
  "/api/satellites/S1A/events": [detected, falseAlarm],
  "/api/events/S1A-D100": {
    ...detected, satellite: "S1A", esa_type: "station_keeping", esa_da_m: 9.4, di_mdeg: 0.01,
    de_1e6: 0.1, statistic: 8, channel: "a", alarm_delay_revs: 1, split: "test",
  },
  "/api/robustness": {
    axes: { sigma_m: [0.01, 1000], dv_cm_s: [0.1, 10], points_per_day: [1, 8640], rho: [0, 0.95] },
    p_detect: [[[[0, 0], [0.5, 0.4]], [[0.2, 0.1], [1, 1]]],
               [[[0, 0], [0, 0]], [[0, 0], [0.3, 0.2]]]],
    p_detect_index_order: ["sigma_m", "dv_cm_s", "points_per_day", "rho"],
    min_dv_90_cm_s: [[[null, null], [0.2, 0.3]], [[null, null], [null, null]]],
    presets: { pod: { label: "POD Copernicus", sigma_m: 0.03, points_per_day: 8640, rho: 0 },
               tle: { label: "Type TLE", sigma_m: 1000, points_per_day: 2, rho: 0.9 } },
    detector: fixture.detector, trials_per_cell: 60, window_days: 10,
  },
  "/api/lab/S1A-D100": labWindow("S1A-D100",
                                  [{ start: eventIso, dv_t_mm_s: 5.0, type: "station_keeping" }]),
  // Same real states (they contain a step) but no logged manoeuvre: a false alarm.
  "/api/lab/S1A-F200": labWindow("S1A-F200", []),
  "/api/events/S1A-F200": {
    ...falseAlarm, satellite: "S1A", esa_type: null, esa_da_m: null, di_mdeg: 0.0, de_1e6: 0.0,
    statistic: 7, channel: "a", alarm_delay_revs: 1, split: "test",
  },
};

class InThreadWorker {
  onmessage: ((e: MessageEvent) => void) | null = null;
  private prepared: ReturnType<typeof prepare> | null = null;

  postMessage(msg: WorkerRequest) {
    setTimeout(() => {
      if (msg.type === "load") {
        this.prepared = prepare(msg.window, msg.realDvMmS, msg.config);
        this.onmessage?.({ data: { type: "loaded", realDvMmS: this.prepared.realDvMmS } } as MessageEvent);
      } else if (this.prepared) {
        const result = runLab(this.prepared, msg.params);
        this.onmessage?.({ data: { type: "result", id: msg.id, result, ms: 1 } } as MessageEvent);
      }
    }, 0);
  }

  terminate() {}
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const body = payloads[url];
    return new Response(JSON.stringify(body ?? { detail: "not found" }),
                        { status: body ? 200 : 404 });
  }));
  vi.stubGlobal("Worker", InThreadWorker);
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0));
  vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
});
afterEach(() => vi.unstubAllGlobals());

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  return render(<App client={client} router={router} />);
}

describe("Mission screen", () => {
  it("shows the yearly counters, markers and the event card", async () => {
    renderAt("/?annee=2025");
    expect(await screen.findByRole("heading", { name: /Journal orbital/ })).toBeInTheDocument();
    const stats = screen.getByRole("region", { name: "Bilan 2025" });
    // Counters come from the events of the displayed range (1 detected + 1 missed).
    expect(await within(stats).findByText("2")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /Manœuvre détectée, 1 mars 2025/ }))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Manœuvre manquée/ })).toBeInTheDocument();
    expect(await screen.findByText("Maintien à poste")).toBeInTheDocument();
    expect(screen.getByText("5,2 mm/s")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Tester dans le labo/ }))
      .toHaveAttribute("href", "/labo/S1A-D100");
    expect(screen.getByText(/Contains modified Copernicus Sentinel data/)).toBeInTheDocument();
  });
});

describe("Lab screen", () => {
  it("runs the worker, shows a verdict, the key figure and the heatmap", async () => {
    renderAt("/labo/S1A-D100");
    expect(await screen.findByRole("heading", { name: /Jusqu’où voit-on/ })).toBeInTheDocument();
    expect(await screen.findByText("Manœuvre détectée", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByLabelText("Bruit de position")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /Probabilité de détection/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "POD Copernicus" }))
      .toHaveAttribute("aria-pressed", "true");
  });
});

describe("Lab screen, other cases", () => {
  it("opens a false alarm with no manoeuvre and reports the alarm", async () => {
    renderAt("/labo/S1A-F200");
    expect(await screen.findByText("Fausse alarme", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.getByLabelText("Taille de la manœuvre")).toHaveValue("0");
    expect(screen.queryByText("Manœuvre détectée")).not.toBeInTheDocument();
  });

  it("resets the manoeuvre size when switching to another event", async () => {
    renderAt("/labo/S1A-F200");
    expect(await screen.findByText("Fausse alarme", {}, { timeout: 5000 })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "S1A-D100" } });
    expect(await screen.findByText("Manœuvre détectée", {}, { timeout: 5000 })).toBeInTheDocument();
    await waitFor(() =>
      expect(document.querySelector("output[for=dv]")?.textContent).toBe("5,0 mm/s"));
  });
});
