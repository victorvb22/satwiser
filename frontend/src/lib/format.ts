/** French number and date formatting used across the UI. */

const MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                "septembre", "octobre", "novembre", "décembre"];
export const MONTHS_SHORT = ["JAN", "FÉV", "MAR", "AVR", "MAI", "JUIN", "JUIL", "AOÛT", "SEPT",
                             "OCT", "NOV", "DÉC"];

export function num(value: number, digits = 1): string {
  return value.toLocaleString("fr-FR", { minimumFractionDigits: digits,
                                          maximumFractionDigits: digits });
}

export function signed(value: number, digits = 1): string {
  return (value > 0 ? "+" : value < 0 ? "−" : "") + num(Math.abs(value), digits);
}

/** Δv given in mm/s, shown in mm/s below 1 cm/s and cm/s above. */
export function dv(mmPerS: number | null | undefined): string {
  if (mmPerS === null || mmPerS === undefined || Number.isNaN(mmPerS)) return "—";
  const a = Math.abs(mmPerS);
  if (a < 10) return `${num(a, a < 1 ? 2 : 1)} mm/s`;
  if (a < 1000) return `${num(a / 10, a < 100 ? 1 : 0)} cm/s`;
  return `${num(a / 1000, 2)} m/s`;
}

/** Length in metres with an adapted unit. */
export function length(m: number | null | undefined, digits?: number): string {
  if (m === null || m === undefined || Number.isNaN(m)) return "—";
  const a = Math.abs(m);
  if (a < 1) return `${num(a * 100, digits ?? (a < 0.1 ? 1 : 0))} cm`;
  if (a < 1000) return `${num(a, digits ?? (a < 10 ? 1 : 0))} m`;
  return `${num(a / 1000, digits ?? 1)} km`;
}

export function dateFr(iso: string): string {
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

export function timeFr(iso: string): string {
  const d = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
  return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")} UTC`;
}

export function pct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${num(value * 100, digits)} %`;
}

export const CLASS_LABEL: Record<string, string> = {
  station_keeping: "Maintien à poste",
  orbit_change: "Changement d’orbite",
  unexplained: "Détection inexpliquée",
};

export const ESA_TYPE_LABEL: Record<string, string> = {
  station_keeping: "maintien à poste",
  inclination: "inclinaison",
  sequence: "séquence multi-poussées",
  lowering: "abaissement",
};

export const KIND_LABEL: Record<string, string> = {
  detected: "Manœuvre détectée",
  missed: "Manœuvre manquée",
  false_alarm: "Fausse alarme",
  unscored: "Détection hors évaluation",
};
