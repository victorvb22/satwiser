import { useQueries, useQueryClient } from "@tanstack/react-query";
import { scaleLinear } from "d3-scale";
import { area, line } from "d3-shape";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  queries,
  useEvent,
  useOverview,
  useSatellites,
  type EventSummary,
  type Satellite,
} from "../api/client";
import { ErrorNote, Loading } from "../components/Layout";
import { CLASS_LABEL, dateFr, dv, ESA_TYPE_LABEL, KIND_LABEL, MONTHS_SHORT, num, signed,
         timeFr } from "../lib/format";
import { lttb } from "../lib/lttb";

const W = 936;
const H = 340;
const SOLAR_H = 44;
const OVERVIEW_H = 36;
const R_EARTH = 6378137.0;
const DAY = 86_400_000;
const MIN_SPAN = 5 * DAY;
const EDGE_PX = 10;

type Range = [number, number];

interface Merged {
  orbit: number[];
  t: number[];
  a: (number | null)[];
  f107: (number | null)[];
}

// ------------------------------------------------------------------------- helpers

function yearRange(year: number): Range {
  return [Date.UTC(year, 0, 1), Date.UTC(year + 1, 0, 1)];
}

function exactYear([t0, t1]: Range): number | null {
  const y = new Date(t0).getUTCFullYear();
  const [y0, y1] = yearRange(y);
  return t0 === y0 && t1 === y1 ? y : null;
}

function isoDay(t: number): string {
  return new Date(t).toISOString().slice(0, 10);
}

function quantile(sorted: number[], q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
}

/** Symmetric axis span covering the bulk of the values (transient spikes are clipped). */
function niceSpan(values: number[]): number {
  const sorted = [...values].sort((p, q) => p - q);
  const max = sorted.length
    ? Math.max(Math.abs(quantile(sorted, 0.01)), Math.abs(quantile(sorted, 0.99)), 1) : 1;
  const steps = [5, 10, 20, 25, 50, 100, 150, 200, 250, 300, 500, 1000];
  return steps.find((s) => s >= max * 1.08) ?? Math.ceil(max / 500) * 500;
}

/** Tick marks adapted to the displayed span: days, months or years. */
function ticks([t0, t1]: Range): { t: number; label: string }[] {
  const span = t1 - t0;
  const out: { t: number; label: string }[] = [];
  if (span <= 60 * DAY) {
    const step = [1, 2, 5, 7, 10].find((s) => span / (s * DAY) <= 12) ?? 10;
    const d = new Date(t0);
    let t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
    if (t < t0) t += DAY;
    for (; t <= t1; t += step * DAY) {
      const dt = new Date(t);
      out.push({ t, label: `${dt.getUTCDate()} ${MONTHS_SHORT[dt.getUTCMonth()]}` });
    }
  } else if (span <= 3.2 * 365 * DAY) {
    const step = [1, 2, 3, 6].find((s) => span / (s * 30.4 * DAY) <= 13) ?? 6;
    const d = new Date(t0);
    let y = d.getUTCFullYear();
    let m = Math.ceil((d.getUTCMonth() + (d.getUTCDate() > 1 ? 1 : 0)) / step) * step;
    for (;;) {
      if (m >= 12) { y += Math.floor(m / 12); m %= 12; }
      const t = Date.UTC(y, m, 1);
      if (t > t1) break;
      if (t >= t0) out.push({ t, label: m === 0 ? String(y) : MONTHS_SHORT[m] });
      m += step;
    }
  } else {
    const step = span > 8 * 365 * DAY ? 2 : 1;
    for (let y = new Date(t0).getUTCFullYear(); ; y++) {
      const t = Date.UTC(y, 0, 1);
      if (t > t1) break;
      if (t >= t0 && y % step === 0) out.push({ t, label: String(y) });
    }
  }
  return out;
}

function rangeLabel(range: Range): string {
  const y = exactYear(range);
  return y !== null ? String(y) : `${dateFr(isoDay(range[0]))} → ${dateFr(isoDay(range[1] - 1))}`;
}

// ---------------------------------------------------------------------- data access

/** Series and events for an arbitrary range, assembled from the per-year endpoints. */
function useRangeData(sat: Satellite, [t0, t1]: Range) {
  const years = sat.years.filter((y) => {
    const [y0, y1] = yearRange(y);
    return y1 > t0 && y0 < t1;
  });
  const seriesQs = useQueries({ queries: years.map((y) => queries.series(sat.id, y)) });
  const eventQs = useQueries({ queries: years.map((y) => queries.events(sat.id, y)) });
  const seriesReady = seriesQs.every((q) => q.data);
  const eventsReady = eventQs.every((q) => q.data);
  const seriesStamp = seriesQs.map((q) => q.dataUpdatedAt).join();
  const eventStamp = eventQs.map((q) => q.dataUpdatedAt).join();

  const merged = useMemo<Merged | null>(() => {
    if (!seriesReady) return null;
    const out: Merged = { orbit: [], t: [], a: [], f107: [] };
    for (const q of seriesQs) {
      const s = q.data!;
      for (let k = 0; k < s.t_ms.length; k++) {
        if (s.t_ms[k] < t0 || s.t_ms[k] >= t1) continue;
        out.orbit.push(s.orbit[k]);
        out.t.push(s.t_ms[k]);
        out.a.push(s.a_m[k]);
        out.f107.push(s.f107[k]);
      }
    }
    return out;
  }, [seriesReady, seriesStamp, t0, t1]);

  const events = useMemo<EventSummary[] | null>(() => {
    if (!eventsReady) return null;
    return eventQs.flatMap((q) => q.data!).filter((e) => {
      const t = Date.parse(e.time);
      return e.kind !== "unscored" && t >= t0 && t < t1;
    });
  }, [eventsReady, eventStamp, t0, t1]);

  return { series: merged, events, error: seriesQs.some((q) => q.isError), years };
}

// --------------------------------------------------------------------------- charts

interface ChartProps {
  range: Range;
  series: Merged;
  events: EventSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
  animate: string;
}

function SeriesChart({ range, series, events, selected, onSelect, animate }: ChartProps) {
  const x = scaleLinear().domain(range).range([0, W]);
  const valid = series.a.filter((v): v is number => v !== null);
  const centre = valid.length ? [...valid].sort((p, q) => p - q)[Math.floor(valid.length / 2)] : 0;
  const rel = series.a.map((v) => (v === null ? null : v - centre));
  const span = niceSpan(rel.filter((v): v is number => v !== null));
  const y = scaleLinear().domain([-span, span]).range([H - 6, 6]);
  const reduced = lttb(series.t, rel, 1600);
  const path = line<number>().x((_, k) => x(reduced.x[k])).y((v) => y(v))(reduced.y) ?? "";
  const byOrbit = new Map(series.orbit.map((o, k) => [o, rel[k]]));
  const valueNear = (orbit: number, after: boolean) => {
    for (let d = 0; d < 8; d++) {
      const v = byOrbit.get(orbit + (after ? d + 2 : -d));
      if (v !== undefined && v !== null) return v;
    }
    return 0;
  };
  const marks = ticks(range);
  let grid = "";
  for (const m of marks) grid += `M${x(m.t).toFixed(1)} 0V${H}`;
  grid += `M0 ${y(span / 2).toFixed(1)}H${W}M0 ${y(-span / 2).toFixed(1)}H${W}`;

  return (
    <div className="chart-stack" style={{ gap: 14 }}>
      <div className="chart-frame fade-in" key={animate}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img"
             aria-label={`Demi-grand axe moyen par révolution, ${rangeLabel(range)}`}>
          <defs><clipPath id="plot-area"><rect x="0" y="0" width={W} height={H} /></clipPath></defs>
          <path d={grid} stroke="var(--grid)" strokeWidth="1" fill="none" />
          <path d={`M0 ${y(0)}H${W}`} stroke="#232B40" strokeDasharray="2 6" fill="none" />
          <g clipPath="url(#plot-area)">
            <path d={path} stroke="var(--data)" strokeWidth="7" strokeOpacity="0.1" fill="none"
                  strokeLinejoin="round" />
            <path d={path} stroke="var(--data)" strokeWidth="1.3" fill="none"
                  strokeLinejoin="round" />
          </g>
        </svg>
        <span className="axis-label" style={{ left: 0, top: 0 }}>+{num(span, 0)} m</span>
        <span className="axis-label" style={{ left: 0, bottom: 0 }}>−{num(span, 0)} m</span>
        {events.map((e) => {
          const t = Date.parse(e.time);
          const v = valueNear(e.orbit, e.kind === "detected" || e.kind === "missed");
          return (
            <button key={e.id} className={`marker ${e.kind}`} aria-pressed={e.id === selected}
                    aria-label={`${KIND_LABEL[e.kind]}, ${dateFr(e.time)}`}
                    style={{ left: `${(x(t) / W) * 100}%`,
                             top: `${(Math.min(Math.max(y(v), 4), H - 4) / H) * 100}%` }}
                    onClick={() => onSelect(e.id)}>
              <span />
            </button>
          );
        })}
      </div>
      <div className="ticks" aria-hidden="true">
        {marks.map((m) => (
          <span key={m.t} style={{ left: `${(x(m.t) / W) * 100}%` }}>{m.label}</span>
        ))}
      </div>
    </div>
  );
}

function SolarBand({ range, series }: { range: Range; series: Merged }) {
  const x = scaleLinear().domain(range).range([0, W]);
  const y = scaleLinear().domain([50, 300]).range([SOLAR_H, 2]).clamp(true);
  const pts = series.t.map((t, k) => [t, series.f107[k]] as const)
    .filter((p): p is readonly [number, number] => p[1] !== null);
  const reduced = pts.length > 2000 ? pts.filter((_, k) => k % Math.ceil(pts.length / 2000) === 0) : pts;
  const l = line<readonly [number, number]>().x((p) => x(p[0])).y((p) => y(p[1]))(reduced) ?? "";
  const a = area<readonly [number, number]>().x((p) => x(p[0])).y0(SOLAR_H)
    .y1((p) => y(p[1]))(reduced) ?? "";
  return (
    <svg viewBox={`0 0 ${W} ${SOLAR_H}`} aria-hidden="true" style={{ width: "100%", height: "auto" }}>
      <path d={a} fill="var(--event)" fillOpacity="0.1" />
      <path d={l} stroke="var(--event)" strokeOpacity="0.55" fill="none" />
    </svg>
  );
}

type DragMode = "move" | "left" | "right";

interface OverviewProps {
  sat: Satellite;
  range: Range;
  onDraft: (r: Range | null) => void;
  onCommit: (r: Range) => void;
}

/**
 * Whole-mission strip with the displayed window. Drag the window to pan, drag either
 * edge to zoom, click elsewhere to centre the window there. Keyboard: arrows pan,
 * Shift + arrows resize, Home/End jump to the mission ends.
 */
function OverviewStrip({ sat, range, onDraft, onCommit }: OverviewProps) {
  const { data } = useOverview(sat.id);
  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<{ mode: DragMode; start: Range; t: number } | null>(null);
  const draft = useRef<Range | null>(null);
  const first = Date.UTC(sat.years[0], 0, 1);
  const last = Date.UTC(sat.years[sat.years.length - 1] + 1, 0, 1);
  const x = scaleLinear().domain([first, last]).range([0, W]);
  const path = useMemo(() => {
    if (!data) return "";
    const reduced = lttb(data.t_ms, data.a_m, 900);
    const vals = reduced.y;
    // Scale on the operational phase: orbit acquisition (2014) sits kilometres away.
    const opsStart = Date.parse(`${sat.operational_start}T00:00:00Z`);
    const sorted = vals.filter((_, k) => reduced.x[k] >= opsStart).sort((p, q) => p - q);
    const lo = quantile(sorted, 0.02), hi = quantile(sorted, 0.98);
    const y = scaleLinear().domain([lo, hi]).range([OVERVIEW_H - 3, 3]).clamp(true);
    return line<number>().x((_, k) => x(reduced.x[k])).y((v) => y(v))(vals) ?? "";
  }, [data, x, sat.operational_start]);

  const clamp = ([t0, t1]: Range): Range => {
    const span = Math.min(Math.max(t1 - t0, MIN_SPAN), last - first);
    let a = t0, b = t0 + span;
    if (a < first) { a = first; b = first + span; }
    if (b > last) { b = last; a = last - span; }
    return [Math.round(a), Math.round(b)];
  };
  const timeAt = (clientX: number) => {
    const box = ref.current!.getBoundingClientRect();
    return x.invert(((clientX - box.left) / box.width) * W);
  };
  const pxPerMs = () => ref.current!.getBoundingClientRect().width / (last - first);
  const update = (r: Range) => { draft.current = r; onDraft(r); };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const t = timeAt(e.clientX);
    const tol = EDGE_PX / pxPerMs();
    const [t0, t1] = range;
    let mode: DragMode = "move";
    let start = range;
    if (Math.abs(t - t0) <= tol) mode = "left";
    else if (Math.abs(t - t1) <= tol) mode = "right";
    else if (t < t0 || t > t1) {
      start = clamp([t - (t1 - t0) / 2, t + (t1 - t0) / 2]);
      update(start);
    }
    drag.current = { mode, start, t };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.current) {
      const t = timeAt(e.clientX);
      const tol = EDGE_PX / pxPerMs();
      const edge = Math.abs(t - range[0]) <= tol || Math.abs(t - range[1]) <= tol;
      e.currentTarget.style.cursor = edge ? "ew-resize" : t >= range[0] && t <= range[1] ? "grab" : "pointer";
      return;
    }
    const { mode, start, t } = drag.current;
    const dt = timeAt(e.clientX) - t;
    if (mode === "move") update(clamp([start[0] + dt, start[1] + dt]));
    else if (mode === "left") update([Math.max(first, Math.min(start[0] + dt, start[1] - MIN_SPAN)), start[1]]);
    else update([start[0], Math.min(last, Math.max(start[1] + dt, start[0] + MIN_SPAN))]);
  };
  const onPointerUp = () => {
    if (drag.current && draft.current) onCommit(draft.current);
    drag.current = null;
    draft.current = null;
    onDraft(null);
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    const [t0, t1] = range;
    const span = t1 - t0;
    const step = Math.max(span * 0.1, DAY);
    let next: Range | null = null;
    if (e.key === "ArrowRight") next = e.shiftKey ? [t0, t1 + step] : [t0 + step, t1 + step];
    if (e.key === "ArrowLeft") next = e.shiftKey ? [t0, Math.max(t0 + MIN_SPAN, t1 - step)]
                                                 : [t0 - step, t1 - step];
    if (e.key === "Home") next = [first, first + span];
    if (e.key === "End") next = [last - span, last];
    if (next) {
      e.preventDefault();
      onCommit(clamp(next));
    }
  };

  return (
    <div className="overview" ref={ref} role="slider" tabIndex={0}
         aria-label="Fenêtre affichée dans la mission complète (glisser pour déplacer, tirer les bords pour zoomer)"
         aria-valuemin={first} aria-valuemax={last} aria-valuenow={range[0]}
         aria-valuetext={rangeLabel(range)}
         onKeyDown={onKeyDown} onPointerDown={onPointerDown} onPointerMove={onPointerMove}
         onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
      <svg viewBox={`0 0 ${W} ${OVERVIEW_H}`} aria-hidden="true"
           style={{ width: "100%", height: "auto", display: "block" }}>
        <path d={path} stroke="var(--faint)" strokeWidth="0.6" fill="none" />
      </svg>
      <span className="overview-frame"
            style={{ left: `${(x(range[0]) / W) * 100}%`,
                     width: `${((x(range[1]) - x(range[0])) / W) * 100}%` }}>
        <span className="handle left" />
        <span className="handle right" />
      </span>
    </div>
  );
}

// ------------------------------------------------------------------------ event card

function EventCard({ id, satellite }: { id: string | null; satellite: string }) {
  const { data: e, isLoading } = useEvent(id);
  if (!id) return <div className="card"><span className="muted">Aucun événement sur cette période.</span></div>;
  if (isLoading || !e) return <div className="card skeleton" style={{ height: 330 }} />;
  const title = e.kind === "missed" ? "Manœuvre manquée"
    : e.kind === "false_alarm" ? "Fausse alarme"
    : CLASS_LABEL[e.class ?? ""] ?? "Détection";
  const status = { detected: "DÉTECTÉE", missed: "MANQUÉE", false_alarm: "NON ÉTIQUETÉE",
                   unscored: "HORS ÉVALUATION" }[e.kind];
  const orange = e.kind === "detected" || e.kind === "missed";
  return (
    <div className="card fade-in" key={e.id}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span className="kicker">ÉVÉNEMENT</span>
        <span className="tag" style={{ color: orange ? "var(--event)" : "var(--muted)",
                                       borderColor: orange ? "rgba(244,162,89,0.4)" : "#2A3148" }}>
          {status}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontWeight: 300, fontSize: 28, lineHeight: 1.1, letterSpacing: -0.8 }}>{title}</span>
        <span className="muted" style={{ fontSize: 14 }}>{dateFr(e.time)} · {timeFr(e.time)}</span>
        {e.kind === "false_alarm" && e.class && (
          <span className="faint" style={{ fontSize: 13 }}>
            Aucune manœuvre ESA à ±1 révolution. Signature : {CLASS_LABEL[e.class].toLowerCase()}.
          </span>
        )}
      </div>
      <div className="divider">
        <div className="card-row"><span>Δv estimé</span><span>{dv(e.dv_est_mm_s)}</span></div>
        <div className="card-row"><span>Δv ESA</span><span>{dv(e.dv_esa_mm_s)}</span></div>
        <div className="card-row">
          <span>Saut de demi-grand axe</span>
          <span>{e.da_m === null ? "—" : `${signed(e.da_m, Math.abs(e.da_m) < 10 ? 1 : 0)} m`}</span>
        </div>
        <div className="card-row"><span>Saut d’inclinaison</span>
          <span>{e.di_mdeg === null ? "—" : `${signed(e.di_mdeg, 2)} mdeg`}</span></div>
        <div className="card-row"><span>Statistique CUSUM</span>
          <span>{e.statistic === null ? "—" : num(e.statistic, 1)}</span></div>
        {e.esa_type && (
          <div className="card-row"><span>Type ESA</span><span>{ESA_TYPE_LABEL[e.esa_type] ?? e.esa_type}</span></div>
        )}
      </div>
      {e.lab_available ? (
        <Link className="button-outline" to={`/labo/${e.id}`}>Tester dans le labo →</Link>
      ) : (
        <Link className="button-outline" to="/labo" title={`Fenêtre non préchargée pour ${satellite}`}>
          Labo (événement par défaut) →
        </Link>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------- page

function MissionView({ sat }: { sat: Satellite }) {
  const [params, setParams] = useSearchParams();
  const client = useQueryClient();
  const defaultYear = sat.years[sat.years.length - 2] ?? sat.years[0];
  const committed = useMemo<Range>(() => {
    const d = Date.parse(`${params.get("debut")}T00:00:00Z`);
    const f = Date.parse(`${params.get("fin")}T00:00:00Z`);
    if (Number.isFinite(d) && Number.isFinite(f) && f - d >= MIN_SPAN) return [d, f];
    return yearRange(Number(params.get("annee")) || defaultYear);
  }, [params, defaultYear]);
  const [draft, setDraft] = useState<Range | null>(null);
  const range = draft ?? committed;
  const commit = (r: Range) => {
    const y = exactYear(r);
    setParams(y !== null ? { annee: String(y) } : { debut: isoDay(r[0]), fin: isoDay(r[1]) },
              { replace: true });
  };
  const { series, events, error, years } = useRangeData(sat, range);
  const [selected, setSelected] = useState<string | null>(null);
  const shown = useMemo(() => events ?? [], [events]);

  useEffect(() => {
    if (draft) return;
    if (!shown.length) return setSelected(null);
    if (selected && shown.some((e) => e.id === selected)) return;
    const preferred = shown.find((e) => e.id === sat.default_lab_event)
      ?? shown.find((e) => e.kind === "detected") ?? shown[0];
    setSelected(preferred.id);
  }, [shown, selected, sat.default_lab_event, draft]);

  useEffect(() => {
    const lo = Math.min(...years) - 1;
    const hi = Math.max(...years) + 1;
    for (const y of [lo, hi]) {
      if (sat.years.includes(y)) {
        void client.prefetchQuery(queries.series(sat.id, y));
        void client.prefetchQuery(queries.events(sat.id, y));
      }
    }
  }, [client, sat, years]);

  const counts = useMemo(() => {
    if (!events) return null;
    const detected = events.filter((e) => e.kind === "detected").length;
    const missed = events.filter((e) => e.kind === "missed").length;
    return { esa: detected + missed, detected, missed,
             fa: events.filter((e) => e.kind === "false_alarm").length };
  }, [events]);
  const meanA = series ? series.a.filter((v): v is number => v !== null) : [];
  const altitude = meanA.length ? meanA.reduce((s, v) => s + v, 0) / meanA.length - R_EARTH : null;
  const year = exactYear(committed);
  const opsStart = Date.parse(`${sat.operational_start}T00:00:00Z`);

  return (
    <>
      <header className="page-header">
        <div className="intro">
          <span className="eyebrow">
            {sat.name.toUpperCase()} · HÉLIOSYNCHRONE
            {altitude !== null && ` · ${num(altitude / 1000, 0)} KM`}
          </span>
          <h1 className="title">Journal <em>orbital</em></h1>
          <p className="lede">
            Demi-grand axe moyen par révolution, signature de la trace au sol retirée. La
            traînée fait descendre l’orbite, chaque manœuvre la remonte.
          </p>
        </div>
        <div className="pills" role="group" aria-label="Année affichée">
          {sat.years.map((y) => (
            <button key={y} className="pill" aria-pressed={y === year} onClick={() => commit(yearRange(y))}>
              {y}
            </button>
          ))}
        </div>
      </header>

      <section className="stats" aria-label={`Bilan ${rangeLabel(range)}`}>
        <div className="stat"><span className="label">Manœuvres étiquetées ESA</span>
          <span className="value">{counts?.esa ?? "—"}</span></div>
        <div className="stat"><span className="label">Détectées</span>
          <span className="value" style={{ color: "var(--event)" }}>{counts?.detected ?? "—"}</span></div>
        <div className="stat"><span className="label">Manquées</span>
          <span className="value">{counts?.missed ?? "—"}</span></div>
        <div className="stat"><span className="label">Fausses alarmes</span>
          <span className="value muted">{counts?.fa ?? "—"}</span></div>
      </section>

      <div className="mission-body">
        <div className="chart-stack">
          <div className="label" style={{ display: "flex", justifyContent: "space-between" }}>
            <span>{rangeLabel(range)}</span>
            {year === null && (
              <button className="link-button" onClick={() => commit(yearRange(new Date(range[0]).getUTCFullYear()))}>
                Revenir à l’année
              </button>
            )}
          </div>
          {series && events ? (
            <SeriesChart range={range} series={series} events={shown} selected={selected}
                         onSelect={setSelected} animate={`${committed[0]}-${committed[1]}`} />
          ) : error ? (
            <ErrorNote>Série indisponible pour cette période.</ErrorNote>
          ) : (
            <div className="skeleton" style={{ aspectRatio: `${W} / ${H}` }} />
          )}
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            <span className="label">Activité solaire · F10.7</span>
            {series && <SolarBand range={range} series={series} />}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
            <span className="label">
              Mission complète · {sat.years[0]} → {sat.years[sat.years.length - 1]} · glisser la
              fenêtre pour se déplacer, tirer ses bords pour zoomer
            </span>
            <OverviewStrip sat={sat} range={range} onDraft={setDraft} onCommit={commit} />
          </div>
          {range[0] < opsStart && (
            <p className="faint" style={{ fontSize: 12, margin: 0 }}>
              Avant août 2014 : acquisition de l’orbite de référence, hors évaluation.
            </p>
          )}
        </div>
        <aside style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <EventCard id={selected} satellite={sat.name} />
          <div className="legend">
            <div><span className="marker detected" style={{ position: "static", margin: 0, width: 9 }}><span /></span>Manœuvre détectée</div>
            <div><span className="marker missed" style={{ position: "static", margin: 0, width: 9 }}><span /></span>Manœuvre manquée</div>
            <div><span className="marker false_alarm" style={{ position: "static", margin: 0, width: 9 }}><span /></span>Fausse alarme</div>
            <p className="faint" style={{ margin: 0, fontSize: 12, lineHeight: 1.5 }}>
              Détecteur principal (CUSUM), évalué contre l’historique de manœuvres ESA.
              Test : {sat.summary.by_split.test.detected} détectées sur{" "}
              {sat.summary.by_split.test.esa_manoeuvres} depuis 2020.
            </p>
          </div>
        </aside>
      </div>
    </>
  );
}

export default function Mission() {
  const sats = useSatellites();
  if (sats.isError) return <ErrorNote>API injoignable. Réessayez dans un instant.</ErrorNote>;
  if (!sats.data) return <Loading what="de la mission" />;
  return <MissionView sat={sats.data[0]} />;
}
