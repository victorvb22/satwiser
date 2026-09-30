import { useQueryClient } from "@tanstack/react-query";
import { scaleLinear } from "d3-scale";
import { area, line } from "d3-shape";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  queries,
  useEvent,
  useEvents,
  useOverview,
  useSatellites,
  useSeries,
  type EventSummary,
  type Satellite,
  type Series,
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

function yearBounds(year: number): [number, number] {
  return [Date.UTC(year, 0, 1), Date.UTC(year + 1, 0, 1)];
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

interface ChartProps {
  series: Series;
  events: EventSummary[];
  selected: string | null;
  onSelect: (id: string) => void;
}

function SeriesChart({ series, events, selected, onSelect }: ChartProps) {
  const [t0, t1] = yearBounds(series.year);
  const x = scaleLinear().domain([t0, t1]).range([0, W]);
  const valid = series.a_m.filter((v): v is number => v !== null);
  const centre = valid.length ? [...valid].sort((p, q) => p - q)[Math.floor(valid.length / 2)] : 0;
  const rel = series.a_m.map((v) => (v === null ? null : v - centre));
  const span = niceSpan(rel.filter((v): v is number => v !== null));
  const y = scaleLinear().domain([-span, span]).range([H - 6, 6]);
  const reduced = lttb(series.t_ms, rel, 1600);
  const path = line<number>().x((_, k) => x(reduced.x[k])).y((v) => y(v))(reduced.y) ?? "";
  const byOrbit = new Map(series.orbit.map((o, k) => [o, rel[k]]));
  const valueNear = (orbit: number, after: boolean) => {
    for (let d = 0; d < 8; d++) {
      const v = byOrbit.get(orbit + (after ? d + 2 : -d));
      if (v !== undefined && v !== null) return v;
    }
    return 0;
  };
  let grid = "";
  for (let k = 0; k <= 12; k++) grid += `M${(k * W / 12).toFixed(1)} 0V${H}`;
  grid += `M0 ${y(span / 2).toFixed(1)}H${W}M0 ${y(-span / 2).toFixed(1)}H${W}`;

  return (
    <div className="chart-frame fade-in" key={series.year}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Demi-grand axe moyen par révolution en ${series.year}`}>
        <defs><clipPath id="plot-area"><rect x="0" y="0" width={W} height={H} /></clipPath></defs>
        <path d={grid} stroke="var(--grid)" strokeWidth="1" fill="none" />
        <path d={`M0 ${y(0)}H${W}`} stroke="#232B40" strokeDasharray="2 6" fill="none" />
        <g clipPath="url(#plot-area)">
          <path d={path} stroke="var(--data)" strokeWidth="7" strokeOpacity="0.1" fill="none"
                strokeLinejoin="round" />
          <path d={path} stroke="var(--data)" strokeWidth="1.3" fill="none" strokeLinejoin="round" />
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
  );
}

function SolarBand({ series }: { series: Series }) {
  const [t0, t1] = yearBounds(series.year);
  const x = scaleLinear().domain([t0, t1]).range([0, W]);
  const y = scaleLinear().domain([50, 300]).range([SOLAR_H, 2]).clamp(true);
  const pts = series.t_ms.map((t, k) => [t, series.f107[k]] as const)
    .filter((p): p is readonly [number, number] => p[1] !== null);
  const l = line<readonly [number, number]>().x((p) => x(p[0])).y((p) => y(p[1]))(pts) ?? "";
  const a = area<readonly [number, number]>().x((p) => x(p[0])).y0(SOLAR_H).y1((p) => y(p[1]))(pts) ?? "";
  return (
    <svg viewBox={`0 0 ${W} ${SOLAR_H}`} aria-hidden="true" style={{ width: "100%", height: "auto" }}>
      <path d={a} fill="var(--event)" fillOpacity="0.1" />
      <path d={l} stroke="var(--event)" strokeOpacity="0.55" fill="none" />
    </svg>
  );
}

function OverviewStrip({ sat, year, onYear }: { sat: Satellite; year: number;
                                                onYear: (y: number) => void }) {
  const { data } = useOverview(sat.id);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
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
  const pick = (clientX: number) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const t = x.invert(((clientX - box.left) / box.width) * W);
    const y = new Date(t).getUTCFullYear();
    if (sat.years.includes(y) && y !== year) onYear(y);
  };
  const [y0, y1] = yearBounds(year);
  return (
    <div className="overview" ref={ref} role="slider" tabIndex={0}
         aria-label="Année dans la mission complète" aria-valuemin={sat.years[0]}
         aria-valuemax={sat.years[sat.years.length - 1]} aria-valuenow={year}
         onKeyDown={(e) => {
           const k = sat.years.indexOf(year);
           if (e.key === "ArrowRight" && k < sat.years.length - 1) onYear(sat.years[k + 1]);
           if (e.key === "ArrowLeft" && k > 0) onYear(sat.years[k - 1]);
         }}
         onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); pick(e.clientX); }}
         onPointerMove={(e) => { if (dragging.current) pick(e.clientX); }}
         onPointerUp={() => { dragging.current = false; }}>
      <svg viewBox={`0 0 ${W} ${OVERVIEW_H}`} aria-hidden="true" style={{ width: "100%", height: "auto", display: "block" }}>
        <path d={path} stroke="var(--faint)" strokeWidth="0.6" fill="none" />
      </svg>
      <span className="overview-frame"
            style={{ left: `${(x(y0) / W) * 100}%`, width: `${((x(y1) - x(y0)) / W) * 100}%` }} />
    </div>
  );
}

function EventCard({ id, satellite }: { id: string | null; satellite: string }) {
  const { data: e, isLoading } = useEvent(id);
  if (!id) return <div className="card"><span className="muted">Aucun événement cette année.</span></div>;
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

function MissionView({ sat }: { sat: Satellite }) {
  const [params, setParams] = useSearchParams();
  const client = useQueryClient();
  const year = Number(params.get("annee")) || sat.years[sat.years.length - 2] || sat.years[0];
  const setYear = (y: number) => setParams({ annee: String(y) }, { replace: true });
  const series = useSeries(sat.id, year);
  const events = useEvents(sat.id, year);
  const [selected, setSelected] = useState<string | null>(null);
  const shown = useMemo(() => (events.data ?? []).filter((e) => e.kind !== "unscored"), [events.data]);

  useEffect(() => {
    if (!shown.length) return setSelected(null);
    if (selected && shown.some((e) => e.id === selected)) return;
    const preferred = shown.find((e) => e.id === sat.default_lab_event)
      ?? shown.find((e) => e.kind === "detected") ?? shown[0];
    setSelected(preferred.id);
  }, [shown, selected, sat.default_lab_event]);

  useEffect(() => {
    for (const y of [year - 1, year + 1]) {
      if (sat.years.includes(y)) {
        void client.prefetchQuery(queries.series(sat.id, y));
        void client.prefetchQuery(queries.events(sat.id, y));
      }
    }
  }, [client, sat, year]);

  const counts = sat.summary.by_year[String(year)];
  const meanA = series.data ? series.data.a_m.filter((v): v is number => v !== null) : [];
  const altitude = meanA.length ? meanA.reduce((s, v) => s + v, 0) / meanA.length - R_EARTH : null;

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
            <button key={y} className="pill" aria-pressed={y === year} onClick={() => setYear(y)}>
              {y}
            </button>
          ))}
        </div>
      </header>

      <section className="stats" aria-label={`Bilan ${year}`}>
        <div className="stat"><span className="label">Manœuvres étiquetées ESA</span>
          <span className="value">{counts?.esa_manoeuvres ?? "—"}</span></div>
        <div className="stat"><span className="label">Détectées</span>
          <span className="value" style={{ color: "var(--event)" }}>{counts?.detected ?? "—"}</span></div>
        <div className="stat"><span className="label">Manquées</span>
          <span className="value">{counts?.missed ?? "—"}</span></div>
        <div className="stat"><span className="label">Fausses alarmes</span>
          <span className="value muted">{counts?.false_alarms ?? "—"}</span></div>
      </section>

      <div className="mission-body">
        <div className="chart-stack">
          {series.data && events.data ? (
            <SeriesChart series={series.data} events={shown} selected={selected}
                         onSelect={setSelected} />
          ) : series.isError ? (
            <ErrorNote>Série indisponible pour {year}.</ErrorNote>
          ) : (
            <div className="skeleton" style={{ aspectRatio: `${W} / ${H}` }} />
          )}
          <div className="months">{MONTHS_SHORT.map((m) => <span key={m}>{m}</span>)}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            <span className="label">Activité solaire · F10.7</span>
            {series.data && <SolarBand series={series.data} />}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 4 }}>
            <span className="label">
              Mission complète · {sat.years[0]} → {sat.years[sat.years.length - 1]}
            </span>
            <OverviewStrip sat={sat} year={year} onYear={setYear} />
          </div>
          {year === 2014 && (
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
