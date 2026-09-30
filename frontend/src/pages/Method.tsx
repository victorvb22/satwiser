import { useMetrics, useRobustness, useSatellites } from "../api/client";
import { FCD_URL, REPO_URL } from "../components/Layout";
import { dv, length, num, pct } from "../lib/format";
import { minDetectable } from "../lib/grid";

/** French display names for the detectors of the evaluation table. */
function detectorName(raw: string): string {
  const settings = raw.match(/\(([^)]*)\)/)?.[1];
  if (raw.startsWith("Baseline, raw")) return `Référence, série brute${settings ? ` (${settings.replace("step 2, ", "")})` : ""}`;
  if (raw.startsWith("Baseline, template")) return `Référence, série corrigée${settings ? ` (${settings})` : ""}`;
  if (raw.startsWith("Main")) return raw.includes("drag") ? "Principal : gabarit + traînée + CUSUM" : "Principal : gabarit + CUSUM";
  return raw;
}

export default function Method() {
  const metrics = useMetrics();
  const grid = useRobustness();
  const sats = useSatellites();
  const m = metrics.data;
  const g = grid.data;
  const sat = sats.data?.[0];

  return (
    <>
      <header className="page-header">
        <div className="intro">
          <span className="eyebrow">MÉTHODE</span>
          <h1 className="title">Comment <em>ça marche</em></h1>
          <p className="lede">
            Des éphémérides Copernicus à la détection de manœuvres, et jusqu’à la courbe de
            robustesse. Tous les chiffres de cette page sont produits par le dépôt.
          </p>
        </div>
      </header>

      <div className="prose" style={{ marginTop: 40 }}>
        {m && g && (
          <div className="figures">
            <div className="card">
              <span className="label">Rappel · précision (test 2020 → 2026)</span>
              <span className="value">{pct(m.test.recall)} · {pct(m.test.precision)}</span>
              <span className="faint" style={{ fontSize: 12 }}>
                {m.test.tp} manœuvres détectées sur {m.test.observable}, {m.test.fp} fausses alarmes
              </span>
            </div>
            <div className="card">
              <span className="label">Erreur médiane sur le Δv estimé</span>
              <span className="value">{dv(m.test.dv_abs_err_median_mm_s)}</span>
              <span className="faint" style={{ fontSize: 12 }}>
                délai médian de détection {num(m.test.delay_median_h, 1)} h
              </span>
            </div>
            <div className="card">
              <span className="label">Plus petite manœuvre détectée à 90 %</span>
              <span className="value">
                {(() => {
                  const p = g.presets.pod;
                  const v = minDetectable(g, p.sigma_m, p.points_per_day, p.rho);
                  return v === null ? "—" : dv(v * 10);
                })()}
              </span>
              <span className="faint" style={{ fontSize: 12 }}>avec les orbites précises Copernicus</span>
            </div>
          </div>
        )}

        <h2>Le pipeline</h2>
        <ol className="steps">
          <li><span><strong>Orbites précises.</strong> Les fichiers AUX_POEORB de Sentinel-1
            (un vecteur d’état toutes les 10 s, précision centimétrique) sont téléchargés depuis
            le registre open data d’AWS, fusionnés et convertis en éléments orbitaux.</span></li>
          <li><span><strong>Moyenne par révolution.</strong> Le demi-grand axe osculateur
            oscille de ±9 km sous l’effet de J2 ; le terme court-période est retiré
            analytiquement puis la valeur est moyennée sur chaque révolution.</span></li>
          <li><span><strong>Signature de la trace au sol.</strong> Sentinel-1 repasse au-dessus
            du même point tous les 175 tours (12 jours). Le champ de gravité imprime une
            signature qui se répète à l’identique ; un gabarit estimé sur 2014–2019 la retire
            {m && <> et fait passer le bruit d’une révolution à l’autre de {length(m.noise_m.raw)} à{" "}
            {length(m.noise_m.template)}</>}.</span></li>
          <li><span><strong>Traînée et ruptures.</strong> La décroissance due à la traînée est
            modélisée à partir de l’activité solaire (F10.7, Ap) ; un détecteur CUSUM causal sur
            le demi-grand axe et l’inclinaison signale les sauts, dont on déduit le Δv
            (Δv ≈ Δa · v / 2a).</span></li>
          <li><span><strong>Classification.</strong> Chaque détection est rangée en maintien à
            poste, changement d’orbite ou anomalie inexpliquée (aucune manœuvre ESA
            correspondante), par une règle documentée qui a mieux généralisé qu’un modèle
            appris.</span></li>
          <li><span><strong>Dégradation.</strong> Les éphémérides sont dégradées (bruit, cadence,
            corrélation) et la détection est rejouée pour tracer la plus petite manœuvre
            détectable, du POD jusqu’à des données de type TLE.</span></li>
        </ol>

        <h2>Évaluation</h2>
        <p>
          Vérité terrain : l’historique des manœuvres publié par l’ESA. Une détection est
          correcte si elle tombe à une révolution près d’une manœuvre. Réglages choisis sur
          2014–2019 uniquement, résultats rapportés sur 2020–2026 (découpage temporel strict).
        </p>
        {m && (
          <table className="metrics">
            <thead><tr><th>Détecteur</th><th>Rappel</th><th>Précision</th><th>F1</th><th>Délai</th></tr></thead>
            <tbody>
              {m.comparison.map((row) => (
                <tr key={String(row.detector)}>
                  <td>{detectorName(String(row.detector))}</td>
                  <td>{pct(Number(row["test recall"]))}</td>
                  <td>{pct(Number(row["test precision"]))}</td>
                  <td>{num(Number(row["test f1"]), 2)}</td>
                  <td>{num(Number(row["test delay [h]"]), 1)} h</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <h2>Limites connues</h2>
        <ul>
          <li>Le préréglage « Type TLE » est une approximation : un vrai TLE est un jeu
            d’éléments moyens SGP4 dont les erreurs ne sont pas gaussiennes.</li>
          <li>Les réglages ont été calibrés en période de Soleil calme ; au maximum solaire de
            2024, les orages géomagnétiques produisent davantage de fausses alarmes.</li>
          <li>Les labels ESA ne donnent pas de type de manœuvre : il est déduit de la géométrie
            des poussées. Aucune des deux anomalies connues (impact de particule sur
            Sentinel-1A en 2016, panne d’alimentation de Sentinel-1B en 2021) ne laisse de
            signature orbitale mesurable.</li>
          <li>Avec un échantillonnage clairsemé, des termes à courte période non modélisés
            fixent un plancher de détection indépendant du bruit.</li>
          <li>Le labo du navigateur travaille à 1 point par minute (10 s hors ligne).</li>
        </ul>

        <h2>Data governance</h2>
        <p>
          Satwiser n’utilise que des données ouvertes et redistribuables : orbites précises et
          historiques de manœuvres Copernicus Sentinel-1 (ESA), indices F10.7 et Ap du GFZ
          Potsdam (CC BY 4.0). Aucune donnée Space-Track n’est utilisée : son accord
          d’utilisation interdit de transférer données et analyses dérivées à des tiers, et la
          diffusion de ces informations relève du 10 U.S.C. § 2274. Contient des données
          Copernicus Sentinel modifiées (2014–2026).
        </p>

        <h2>Et ensuite</h2>
        <p>
          Détecter une manœuvre → mettre à jour l’orbite → réévaluer le risque de collision.
          Le troisième maillon fait l’objet d’un projet séparé :{" "}
          <a href={FCD_URL} target="_blank" rel="noreferrer">False Calm Detector</a>.
          Code source et rapports reproductibles :{" "}
          <a href={REPO_URL} target="_blank" rel="noreferrer">github.com/victorvb22/satwiser</a>.
          {sat && <> Données du {sat.first.slice(0, 10)} au {sat.last.slice(0, 10)}.</>}
        </p>
      </div>
    </>
  );
}
