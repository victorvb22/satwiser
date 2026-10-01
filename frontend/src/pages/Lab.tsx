import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import {
  useAllEvents,
  useEvent,
  useLabWindow,
  useRobustness,
  useSatellites,
  type Robustness,
} from "../api/client";
import { ErrorNote, Loading } from "../components/Layout";
import { Reveal, useCountUp } from "../components/Motion";
import type { LabConfig, LabParams, LabResult } from "../lab/runLab";
import { useLab } from "../lab/useLab";
import { CLASS_LABEL, dateFr, dv, KIND_LABEL, length, num, pct } from "../lib/format";
import { minDetectable, nearestIndex, probabilityTable } from "../lib/grid";

const W = 908;
const H = 300;
const BROWSER_MAX_POINTS = 1440; // lab windows are shipped at 60 s

// Slider mappings (0-100 positions, logarithmic where the brief asks for it).
const sigmaOf = (v: number) => 0.01 * 10 ** (v / 20); // 1 cm .. 1 km
const sigmaPos = (s: number) => 20 * Math.log10(s / 0.01);
/** Effective cadence: the window is on a 60 s grid, so only 1440 / stride is reachable. */
const pointsOf = (v: number) => {
  const wanted = Math.max(1, Math.round(10 ** ((v / 100) * Math.log10(BROWSER_MAX_POINTS))));
  return BROWSER_MAX_POINTS / Math.max(1, Math.round(BROWSER_MAX_POINTS / wanted));
};
const pointsPos = (p: number) => (100 * Math.log10(p)) / Math.log10(BROWSER_MAX_POINTS);
/** Position 0 removes the manoeuvre; (0, 100] spans 0.2 mm/s .. 1 m/s logarithmically. */
const dvOf = (v: number) => (v <= 0 ? 0 : 0.2 * 10 ** ((v / 100) * Math.log10(5000)));
const dvPos = (mm: number) =>
  mm <= 0 ? 0 : Math.max(0.5, (100 * Math.log10(Math.max(mm, 0.2) / 0.2)) / Math.log10(5000));

const VERDICTS: Record<LabResult["verdict"], [string, string]> = {
  detected: ["Manœuvre détectée", "var(--event)"],
  missed: ["Noyée dans le bruit", "var(--faint)"],
  quiet: ["Rien à signaler", "var(--faint)"],
  false_alarm: ["Fausse alarme", "var(--text)"],
  insufficient: ["Données insuffisantes", "var(--faint)"],
};

interface SliderProps {
  id: string;
  label: string;
  value: number;
  display: string;
  lo: string;
  hi: string;
  max?: number;
  onChange: (v: number) => void;
}

function Slider({ id, label, value, display, lo, hi, max = 100, onChange }: SliderProps) {
  return (
    <div className="slider">
      <div className="slider-head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{display}</output>
      </div>
      <input id={id} type="range" min={0} max={max} step={max === 100 ? 0.5 : 1}
             value={value} onChange={(e) => onChange(Number(e.target.value))}
             aria-valuetext={display} />
      <div className="slider-foot"><span>{lo}</span><span>{hi}</span></div>
    </div>
  );
}

function LabChart({ result }: { result: LabResult }) {
  const x = scaleLinear().domain([0, 10]).range([0, W]);
  const vals = result.means.filter((v): v is number => v !== null);
  const sorted = [...vals].sort((p, q) => p - q);
  const centre = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
  const cleanVals = result.clean.filter((v): v is number => v !== null).map((v) => v - centre);
  const spread = Math.max(...cleanVals.map(Math.abs), ...vals.map((v) => Math.abs(v - centre) * 0.6), 1);
  const y = scaleLinear().domain([-spread * 1.25, spread * 1.25]).range([H - 16, 16]);
  const pts = result.tDays.map((t, k) => [t, result.means[k]] as const)
    .filter((p): p is readonly [number, number] => p[1] !== null && !Number.isNaN(p[0]));
  const cleanPts = result.tDays.map((t, k) => [t, result.clean[k]] as const)
    .filter((p): p is readonly [number, number] => p[1] !== null && !Number.isNaN(p[0]));
  const truth = line<readonly [number, number]>().x((p) => x(p[0])).y((p) => y(p[1] - centre))(cleanPts) ?? "";
  const dotW = pts.length > 200 ? 2.5 : pts.length > 60 ? 4 : 6;
  const dots = pts.map((p) => `M${x(p[0]).toFixed(1)} ${y(Math.max(Math.min(p[1] - centre, spread * 1.24), -spread * 1.24)).toFixed(1)}h0`).join("");
  let grid = "";
  for (let k = 0; k <= 10; k++) grid += `M${x(k).toFixed(1)} 0V${H}`;
  const evX = x(result.eventDays);
  return (
    <div className="chart-frame unfold-in" style={{ borderTop: "1px solid var(--hairline)", borderBottom: "1px solid var(--hairline)" }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Série dégradée et détections">
        <path d={grid} stroke="var(--grid)" fill="none" />
        <path d={`M${evX} 0V${H}`} stroke="var(--faint)" strokeDasharray="3 5" fill="none" />
        <path d={truth} stroke="var(--data)" strokeOpacity="0.35" fill="none" />
        <path d={dots} stroke="var(--data)" strokeWidth={dotW} strokeLinecap="round" fill="none" />
        {result.detections.map((d) => (
          <path key={d.bin} d={`M${x(d.tDays)} 0V${H}`} strokeWidth="1.5" fill="none"
                stroke={d.nearEvent ? "var(--event)" : d.nearOther ? "var(--muted)" : "var(--text)"} />
        ))}
      </svg>
      <span className="axis-label" style={{ top: 10, left: `${(evX / W) * 100 + 0.6}%` }}>MANŒUVRE ÉTUDIÉE</span>
      <span className="axis-label" style={{ right: 0, top: 10 }}>+{length(spread * 1.25)}</span>
      <span className="axis-label" style={{ right: 0, bottom: 10 }}>−{length(spread * 1.25)}</span>
    </div>
  );
}

function cellColour(p: number): string {
  // Dark surface to event orange, with a cream top for certain detection.
  const lerp = (a: number, b: number, t: number) => Math.round(a + (b - a) * t);
  if (p <= 0.9) {
    const t = p / 0.9;
    return `rgb(${lerp(14, 244, t)},${lerp(18, 162, t)},${lerp(34, 89, t)})`;
  }
  const t = (p - 0.9) / 0.1;
  return `rgb(${lerp(244, 252, t)},${lerp(162, 226, t)},${lerp(89, 186, t)})`;
}

function Heatmap({ grid, sigma, dvMm, points, rho }: { grid: Robustness; sigma: number;
                  dvMm: number; points: number; rho: number }) {
  const table = useMemo(() => probabilityTable(grid, points, rho), [grid, points, rho]);
  const ci = nearestIndex(grid.axes.sigma_m, sigma, true);
  const ri = nearestIndex(grid.axes.dv_cm_s, dvMm / 10, true);
  const rows = [...grid.axes.dv_cm_s.keys()].reverse();
  const cols = grid.axes.sigma_m.length;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline",
                    flexWrap: "wrap", gap: "4px 12px", marginTop: 18 }}>
        <span className="label">Carte de robustesse · probabilité de détection</span>
        <span className="axis-label" style={{ position: "static" }}>cadre orange = réglage actuel</span>
      </div>
      <div className="heatmap" role="table" aria-label="Probabilité de détection selon le bruit et le Δv"
           style={{ gridTemplateColumns: `var(--axis-w) repeat(${cols}, minmax(0, 1fr))`, marginTop: 10 }}>
        {rows.map((di, rk) => (
          <div key={di} role="row" style={{ display: "contents" }}>
            <span className="axis" role="rowheader">{dv(grid.axes.dv_cm_s[di] * 10)}</span>
            {grid.axes.sigma_m.map((_, si) => {
              const p = table[si][di];
              const current = si === ci && di === ri;
              return (
                <div key={si} role="cell" className="cell"
                     aria-label={`${dv(grid.axes.dv_cm_s[di] * 10)}, bruit ${length(grid.axes.sigma_m[si])} : ${pct(p)}`}
                     style={{ background: cellColour(p), color: p > 0.55 ? "#06080E" : "#8C95AB",
                              boxShadow: current ? "inset 0 0 0 2px #F4A259, 0 0 0 1px #F4A259" : undefined,
                              // Cascade from the top-left corner (see .reveal.in .cell).
                              ["--d" as string]: `${(rk + si) * 24}ms` }}>
                  {p >= 0.05 ? Math.round(p * 100) : ""}
                </div>
              );
            })}
          </div>
        ))}
        <span />
        {grid.axes.sigma_m.map((s) => <span key={s} className="col-label">{length(s)}</span>)}
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "4px 12px",
                    fontSize: 11, marginTop: 6 }} className="faint">
        <span>Δv (vertical) · bruit de position (horizontal) · {grid.trials_per_cell} essais par case</span>
        <span>moins précis →</span>
      </div>
    </div>
  );
}

function LabView({ eventId, grid }: { eventId: string; grid: Robustness }) {
  const navigate = useNavigate();
  const event = useEvent(eventId);
  const windowQuery = useLabWindow(eventId);
  const allEvents = useAllEvents(event.data?.satellite);
  const labEvents = (allEvents.data ?? []).filter((e) => e.lab_available);
  const byYear = useMemo(() => {
    const groups = new Map<string, typeof labEvents>();
    for (const e of labEvents) {
      const y = e.time.slice(0, 4);
      groups.set(y, [...(groups.get(y) ?? []), e]);
    }
    return [...groups.entries()];
  }, [labEvents]);
  const presets = Object.entries(grid.presets);
  const [preset, setPreset] = useState<string | null>("pod");
  const [sigmaV, setSigmaV] = useState(sigmaPos(grid.presets.pod.sigma_m));
  const [pointsV, setPointsV] = useState(100);
  const [rhoV, setRhoV] = useState(0);
  const [dvV, setDvV] = useState<number | null>(null);
  // The slider sets the magnitude; the sign follows the real manoeuvre (lowering: < 0).
  const [dvSign, setDvSign] = useState(1);
  const [seed, setSeed] = useState(1);

  const realDv = event.data ? (event.data.dv_esa_mm_s ?? (event.data.kind === "false_alarm" ? 0 : null)) : null;
  const config = useMemo<LabConfig>(() => ({
    detector: { windowRevs: grid.detector.window_revs, threshold: grid.detector.threshold,
                normalisation: grid.detector.normalisation, floorM: grid.detector.floor_m },
    template: grid.template_a,
  }), [grid]);
  // Load the window only once the event is known, so the real Δv is never a guess.
  const lab = useLab(event.data ? windowQuery.data : undefined, realDv, useMemo<LabParams>(() => ({
    sigmaM: sigmaOf(sigmaV), pointsPerDay: pointsOf(pointsV), rho: rhoV / 20,
    dvMmS: dvV === null ? 0 : dvSign * dvOf(dvV), seed,
  }), [sigmaV, pointsV, rhoV, dvV, dvSign, seed]), config);
  useEffect(() => {
    if (dvV === null && lab.realDv !== null) {
      setDvSign(lab.realDv < 0 ? -1 : 1);
      setDvV(lab.realDv !== 0 ? dvPos(Math.abs(lab.realDv)) : 0);
    }
  }, [lab.realDv, dvV]);

  const sigma = sigmaOf(sigmaV);
  const points = pointsOf(pointsV);
  const rho = rhoV / 20;
  const dvMm = dvV === null ? 0 : dvOf(dvV);
  const signedDv = (mm: number | null) =>
    mm === null ? "—" : mm === 0 ? "aucune" : `${mm < 0 ? "−" : ""}${dv(Math.abs(mm))}`;
  const minDv = minDetectable(grid, sigma, points, rho);
  // Eased rather than replaced, so that it glides while a slider moves.
  const minDvShown = useCountUp(minDv, 450);
  const choose = (key: string) => {
    const p = grid.presets[key];
    setPreset(key);
    setSigmaV(sigmaPos(p.sigma_m));
    setPointsV(pointsPos(Math.min(p.points_per_day, BROWSER_MAX_POINTS)));
    setRhoV(Math.round(p.rho * 20));
  };
  const touch = (fn: (v: number) => void) => (v: number) => { setPreset(null); fn(v); };
  const result = lab.result;
  const [verdictText, verdictColour] = result ? VERDICTS[result.verdict] : ["Calcul…", "var(--faint)"];
  // The key figure turns grey when the studied manoeuvre is no longer detected; while a
  // new result is computed it keeps its last state.
  const lastDetected = useRef(true);
  if (result) lastDetected.current = result.verdict === "detected";

  return (
    <>
      <header className="page-header" style={{ paddingTop: 44 }}>
        <div className="intro">
          <span className="eyebrow">LABO DE DÉGRADATION</span>
          <h1 className="title" style={{ fontSize: "clamp(40px, 5vw, 60px)" }}>
            Jusqu’où voit-on <em>une manœuvre ?</em>
          </h1>
        </div>
        <div className={`keyfig${lastDetected.current ? "" : " off"}`} aria-live="polite">
          <span className="label">Plus petite manœuvre détectée à 90 %</span>
          <span className="value">{minDvShown === null ? "> 1 m/s" : dv(minDvShown * 10)}</span>
        </div>
      </header>

      <div className="lab-body">
        <Reveal as="aside" className="controls" delay={120}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <span className="kicker">QUALITÉ DES DONNÉES</span>
            <div className="pills" role="group" aria-label="Préréglages">
              {presets.map(([key, p]) => (
                <button key={key} className="pill sans" aria-pressed={preset === key}
                        onClick={() => choose(key)}>{p.label}</button>
              ))}
            </div>
          </div>
          <Slider id="noise" label="Bruit de position" value={sigmaV} display={length(sigma)}
                  lo="1 cm" hi="1 km" onChange={touch(setSigmaV)} />
          <Slider id="points" label="Points par jour" value={pointsV} display={num(points, 0)}
                  lo="1" hi={num(BROWSER_MAX_POINTS, 0)} onChange={touch(setPointsV)} />
          <Slider id="rho" label="Corrélation des erreurs" value={rhoV} max={19}
                  display={num(rho, 2)} lo="0" hi="0,95" onChange={touch(setRhoV)} />
          <Slider id="dv" label="Taille de la manœuvre" value={dvV ?? 0}
                  display={signedDv(dvSign * dvMm)}
                  lo="0 · 0,2 mm/s" hi="1 m/s" onChange={(v) => setDvV(v)} />
          <div className="card" style={{ padding: 20, gap: 12 }}>
            <div className="verdict" aria-live="polite">
              <span className="dot" style={{ background: verdictColour, boxShadow: `0 0 12px ${verdictColour}` }} />
              <span className="text">{verdictText}</span>
            </div>
            <div className="card-row" style={{ fontSize: 13 }}><span>Score z près de la manœuvre</span>
              <span>{result?.zMaxNearEvent == null ? "—" : num(Math.abs(result.zMaxNearEvent), 1)}</span></div>
            <div className="card-row" style={{ fontSize: 13 }}><span>Bruit par point moyenné</span>
              <span>{length(result?.noisePerBinM ?? null)}</span></div>
            <div className="card-row" style={{ fontSize: 13 }}><span>Saut de demi-grand axe</span>
              <span>{result ? `${result.stepM < 0 ? "−" : "+"}${length(Math.abs(result.stepM))}` : "—"}</span></div>
            <div className="card-row" style={{ fontSize: 13 }}><span>Δv réel (ESA)</span>
              <span>{signedDv(lab.realDv)}</span></div>
            <button className="button-outline" onClick={() => setSeed((s) => s + 1)}>Nouveau tirage du bruit</button>
          </div>
        </Reveal>

        {/* On phones the children are reordered (chart first, pinned) by the stylesheet. */}
        <div className="lab-main">
          <div className="lab-chart">
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }} className="label">
              <span>
                {event.data ? `${KIND_LABEL[event.data.kind]} · ${dateFr(event.data.time)}` : "…"}
                {event.data?.class && event.data.kind === "detected" && ` · ${CLASS_LABEL[event.data.class]}`}
              </span>
              <span className="mono">
                {result ? `1 point = ${result.revsPerBin === 1 ? "1 révolution" : `${result.revsPerBin} révolutions`} · ${num(result.nSamples, 0)} états` : ""}
              </span>
            </div>
            {result ? <LabChart result={result} /> : windowQuery.isError
              ? <ErrorNote>Fenêtre indisponible.</ErrorNote>
              : <div className="skeleton" style={{ aspectRatio: `${W} / ${H}` }} />}
            <div className="months"><span>J0</span><span>J2</span><span>J4</span><span>J6</span><span>J8</span><span>J10</span></div>
          </div>
          <label className="label lab-event" style={{ display: "flex", gap: 10, alignItems: "center" }}>
            Événement
            <select value={eventId} onChange={(e) => navigate(`/labo/${e.target.value}`)}
                    style={{ background: "var(--bg)", color: "var(--text)", border: "1px solid var(--border)",
                             borderRadius: 8, padding: "6px 10px", fontFamily: "var(--mono)", fontSize: 12,
                             flex: "0 1 auto", minWidth: 0, maxWidth: "100%" }}>
              {/* While the list loads, show the current event the way the list will. */}
              {labEvents.length === 0 && (
                <option value={eventId}>
                  {event.data
                    ? `${event.data.time.slice(0, 10)} · ${KIND_LABEL[event.data.kind]}${event.data.dv_esa_mm_s ? ` · ${dv(event.data.dv_esa_mm_s)}` : ""}`
                    : "Chargement…"}
                </option>
              )}
              {byYear.map(([year, list]) => (
                <optgroup key={year} label={year}>
                  {list.map((e) => (
                    <option key={e.id} value={e.id}>
                      {e.time.slice(0, 10)} · {KIND_LABEL[e.kind]}
                      {e.dv_esa_mm_s ? ` · ${dv(e.dv_esa_mm_s)}` : ""}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <Reveal variant="fade" className="lab-map">
            <Heatmap grid={grid} sigma={sigma} dvMm={dvMm} points={points} rho={rho} />
          </Reveal>
          <p className="faint lab-note" style={{ fontSize: 12, lineHeight: 1.5, margin: 0 }}>
            La série est recalculée dans le navigateur à partir d’états réels (1 point par minute) :
            dégradation, moyenne par révolution, détecteur. La carte et le chiffre clé viennent
            d’une grille précalculée hors ligne sur des fenêtres calmes (manœuvre isolée).
            « Type TLE » est une approximation gaussienne, pas un vrai TLE.{" "}
            <Link to="/methode">Méthode</Link>
          </p>
        </div>
      </div>
    </>
  );
}

export default function Lab() {
  const { eventId } = useParams();
  const sats = useSatellites();
  const grid = useRobustness();
  if (sats.isError || grid.isError) return <ErrorNote>API injoignable. Réessayez dans un instant.</ErrorNote>;
  if (!sats.data || !grid.data) return <Loading what="du labo" />;
  const id = eventId ?? sats.data[0].default_lab_event;
  // Keyed by event: switching events resets every slider and worker state.
  return <LabView key={id} eventId={id} grid={grid.data} />;
}
