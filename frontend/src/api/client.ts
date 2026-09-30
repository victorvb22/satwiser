/** Typed access to the read-only API, cached with TanStack Query. */
import { useQuery } from "@tanstack/react-query";

const BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export interface Counts {
  esa_manoeuvres: number;
  detected: number;
  missed: number;
  false_alarms: number;
}

export interface Satellite {
  id: string;
  name: string;
  first: string;
  last: string;
  operational_start: string;
  split: string;
  labels_end: string;
  years: number[];
  default_lab_event: string;
  summary: { all: Counts; by_split: Record<string, Counts>; by_year: Record<string, Counts> };
}

export interface Series {
  satellite: string;
  year: number;
  orbit: number[];
  t_ms: number[];
  a_m: (number | null)[];
  f107: (number | null)[];
}

export interface Overview {
  satellite: string;
  t_ms: number[];
  a_m: (number | null)[];
  f107: (number | null)[];
}

export type EventKind = "detected" | "missed" | "false_alarm" | "unscored";
export type EventClass = "station_keeping" | "orbit_change" | "unexplained";

export interface EventSummary {
  id: string;
  kind: EventKind;
  time: string;
  orbit: number;
  class: EventClass | null;
  dv_est_mm_s: number | null;
  dv_esa_mm_s: number | null;
  da_m: number | null;
  lab_available: boolean;
}

export interface EventDetail extends EventSummary {
  satellite: string;
  esa_type: string | null;
  esa_da_m: number | null;
  di_mdeg: number | null;
  de_1e6: number | null;
  statistic: number | null;
  channel: string | null;
  alarm_delay_revs: number | null;
  split: string | null;
}

export interface Robustness {
  axes: { sigma_m: number[]; dv_cm_s: number[]; points_per_day: number[]; rho: number[] };
  /** Indexed [sigma][dv][points][rho]. */
  p_detect: number[][][][];
  p_detect_index_order: string[];
  /** Indexed [sigma][points][rho]; null when 90 % is never reached. */
  min_dv_90_cm_s: (number | null)[][][];
  presets: Record<string, { label: string; sigma_m: number; points_per_day: number; rho: number }>;
  detector: { window_revs: number; threshold: number; normalisation: "window" | "diff";
              floor_m: number };
  trials_per_cell: number;
  window_days: number;
}

export interface LabWindow {
  event_id: string;
  satellite: string;
  start: string;
  step_s: number;
  event_time: string;
  t_offset_s: number[];
  orbit: number[];
  states: Record<"rx" | "ry" | "rz" | "vx" | "vy" | "vz", number[]>;
  template_a: number[];
  detector: Robustness["detector"];
  esa_manoeuvres: { start: string; dv_t_mm_s: number; type: string }[];
}

export interface Metrics {
  satellite: string;
  detector: Record<string, number | string>;
  test: Record<string, number>;
  comparison: Record<string, number | string>[];
  noise_m: { raw: number; template: number };
  source: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) throw new ApiError(response.status, `${response.status} on ${path}`);
  return response.json() as Promise<T>;
}

const DAY = 24 * 3600 * 1000;

export const queries = {
  satellites: () => ({ queryKey: ["satellites"], queryFn: () => getJson<Satellite[]>("/api/satellites"),
                       staleTime: DAY }),
  series: (sat: string, year: number) => ({
    queryKey: ["series", sat, year],
    queryFn: () => getJson<Series>(`/api/satellites/${sat}/series?year=${year}`),
    staleTime: DAY,
  }),
  overview: (sat: string) => ({ queryKey: ["overview", sat],
                                queryFn: () => getJson<Overview>(`/api/satellites/${sat}/overview`),
                                staleTime: DAY }),
  events: (sat: string, year: number) => ({
    queryKey: ["events", sat, year],
    queryFn: () => getJson<EventSummary[]>(`/api/satellites/${sat}/events?year=${year}`),
    staleTime: DAY,
  }),
  allEvents: (sat: string) => ({
    queryKey: ["events", sat, "all"],
    queryFn: () => getJson<EventSummary[]>(`/api/satellites/${sat}/events`),
    staleTime: DAY,
  }),
  event: (id: string) => ({ queryKey: ["event", id],
                            queryFn: () => getJson<EventDetail>(`/api/events/${id}`),
                            staleTime: DAY }),
  robustness: () => ({ queryKey: ["robustness"], queryFn: () => getJson<Robustness>("/api/robustness"),
                       staleTime: DAY }),
  metrics: () => ({ queryKey: ["metrics"], queryFn: () => getJson<Metrics>("/api/metrics"),
                    staleTime: DAY }),
  lab: (id: string) => ({ queryKey: ["lab", id], queryFn: () => getJson<LabWindow>(`/api/lab/${id}`),
                          staleTime: DAY }),
};

export const useSatellites = () => useQuery(queries.satellites());
export const useSeries = (sat: string, year: number) => useQuery(queries.series(sat, year));
export const useOverview = (sat: string) => useQuery(queries.overview(sat));
export const useEvents = (sat: string, year: number) => useQuery(queries.events(sat, year));
export const useAllEvents = (sat: string | undefined) =>
  useQuery({ ...queries.allEvents(sat ?? ""), enabled: sat !== undefined });
export const useEvent = (id: string | null) =>
  useQuery({ ...queries.event(id ?? ""), enabled: id !== null });
export const useRobustness = () => useQuery(queries.robustness());
export const useMetrics = () => useQuery(queries.metrics());
export const useLabWindow = (id: string | null) =>
  useQuery({ ...queries.lab(id ?? ""), enabled: id !== null });
