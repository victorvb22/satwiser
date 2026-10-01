import type { ReactNode } from "react";

import { useMetrics, useRobustness, useSatellites, type Metrics, type Robustness } from "../api/client";
import { FCD_URL, REPO_URL } from "../components/Layout";
import { Reveal } from "../components/Motion";
import { dateFr, dv, length, num, pct } from "../lib/format";
import { minDetectable } from "../lib/grid";

/** French display names for the detectors of the evaluation table. */
function detectorName(raw: string): string {
  const settings = raw.match(/\(([^)]*)\)/)?.[1];
  if (raw.startsWith("Baseline, raw")) {
    return `Référence, série brute${settings ? ` (${settings.replace("step 2, ", "")})` : ""}`;
  }
  if (raw.startsWith("Baseline, template")) {
    return `Référence, série corrigée${settings ? ` (${settings})` : ""}`;
  }
  if (raw.startsWith("Main")) {
    return raw.includes("drag") ? "Principal : gabarit + traînée + CUSUM" : "Principal : gabarit + CUSUM";
  }
  return raw;
}

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <Reveal as="section" className="method-section" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      <div className="method-text">{children}</div>
    </Reveal>
  );
}

function presetFloor(g: Robustness, key: string): string {
  const p = g.presets[key];
  if (!p) return "—";
  const v = minDetectable(g, p.sigma_m, p.points_per_day, p.rho);
  return v === null ? "au-delà de 1 m/s" : dv(v * 10);
}

/** Calibration and test periods as year ranges, from the exported model dates. */
function periods(model: Metrics["model"] | undefined): { cal: string; test: string } {
  if (!model?.coverage_end) return { cal: "la période de calibration", test: "la période de test" };
  const split = Number(model.split.slice(0, 4));
  return {
    cal: `${model.operational_start.slice(0, 4)}–${split - 1}`,
    test: `${split}–${model.coverage_end.slice(0, 4)}`,
  };
}

function Figures({ m, g }: { m: Metrics; g: Robustness }) {
  const test = periods(m.model).test.replace("–", " → ");
  return (
    <div className="figures">
      <Reveal variant="unfold" className="card">
        <span className="label">Rappel · précision (test {test})</span>
        <span className="value">{pct(m.test.recall)} · {pct(m.test.precision)}</span>
        <span className="faint" style={{ fontSize: 12 }}>
          {m.test.tp} manœuvres détectées sur {m.test.observable}, {m.test.fp} fausses alarmes
        </span>
      </Reveal>
      <Reveal variant="unfold" className="card" delay={120}>
        <span className="label">Erreur médiane sur le Δv estimé</span>
        <span className="value">{dv(m.test.dv_abs_err_median_mm_s)}</span>
        <span className="faint" style={{ fontSize: 12 }}>
          délai médian de détection {num(m.test.delay_median_h, 1)} h
        </span>
      </Reveal>
      <Reveal variant="unfold" className="card" delay={240}>
        <span className="label">Plus petite manœuvre détectée à 90 %</span>
        <span className="value">{presetFloor(g, "pod")}</span>
        <span className="faint" style={{ fontSize: 12 }}>avec les orbites précises Copernicus</span>
      </Reveal>
    </div>
  );
}

export default function Method() {
  const metrics = useMetrics();
  const grid = useRobustness();
  const sats = useSatellites();
  const m = metrics.data;
  const g = grid.data;
  const sat = sats.data?.[0];
  const model = m?.model as Metrics["model"] | undefined;
  const beta = model?.drag_beta;
  const period = periods(model);

  return (
    <>
      <header className="page-header">
        <div className="intro">
          <span className="eyebrow">MÉTHODE</span>
          <h1 className="title">Comment <em>ça marche</em></h1>
          <p className="lede" style={{ maxWidth: 760 }}>
            Des éphémérides Copernicus à la détection de manœuvres, et jusqu’à la courbe de
            robustesse. Tous les chiffres de cette page sont calculés par le dépôt et servis
            par l’API ; aucun n’est saisi à la main.
          </p>
        </div>
      </header>

      <div className="method" style={{ marginTop: 40 }}>
        {m && g && <Figures m={m} g={g} />}

        <Section id="objectif" title="Objectif">
          <p>
            Un satellite en orbite basse perd de l’altitude sous l’effet du frottement de la
            haute atmosphère ; ses opérateurs le remontent régulièrement par de petites
            poussées. Pour un tiers qui ne dispose que d’orbites observées (un service de
            surveillance de l’espace, un opérateur voisin), repérer ces manœuvres est
            indispensable : une orbite non mise à jour après une manœuvre fausse toutes les
            prédictions, y compris celles de risque de collision.
          </p>
          <p>
            Satwiser retrouve les manœuvres de Sentinel-1A dans les orbites précises publiées
            par Copernicus, les classe, et mesure jusqu’où la détection tient quand la qualité
            des données se dégrade, des orbites centimétriques jusqu’à des données de précision
            kilométrique comparables à un catalogue public.
          </p>
        </Section>

        <Section id="donnees" title="Données">
          <p>
            <strong>Orbites précises (AUX_POEORB).</strong> Chaque fichier journalier couvre
            26 heures avec un vecteur d’état (position, vitesse) toutes les 10 secondes, exprimé
            dans un repère lié à la Terre ; deux fichiers consécutifs se recouvrent de deux
            heures, et le plus ancien fait foi sur le recouvrement. La mission complète est
            téléchargée depuis le registre open data d’AWS, soit
            {m?.revolutions ? ` ${num(m.revolutions, 0)} révolutions exploitables` : " plusieurs dizaines de milliers de révolutions"}
            {sat && ` du ${dateFr(sat.first)} au ${dateFr(sat.last)}`}.
          </p>
          <p>
            <strong>Historique des manœuvres (ESA).</strong> L’ESA publie, pour chaque poussée,
            ses instants de début et de fin et les trois composantes de l’accélération
            commandée. Le fichier ne précise ni unité, ni repère, ni type de manœuvre : la
            comparaison avec l’orbite montre qu’il s’agit de km/s² dans le repère radial /
            le long de la trace / hors plan. Les poussées séparées de moins de six heures sont
            regroupées en une manœuvre (les écarts entre poussées sont soit inférieurs à cinq
            heures, soit supérieurs à huit), et le type est déduit de la géométrie : maintien à
            poste (poussées le long de la trace, positives), inclinaison (composante hors plan
            dominante), séquence multi-poussées (poussées de signes opposés) ou abaissement.
          </p>
          <p>
            <strong>Activité solaire et géomagnétique.</strong> Flux radio solaire F10.7 et indice
            Ap journaliers du GFZ Potsdam (licence CC BY 4.0), qui pilotent la densité de la
            haute atmosphère et donc la traînée.
          </p>
        </Section>

        <Section id="elements" title="Des vecteurs d’état aux éléments moyens">
          <p>
            Les vecteurs d’état sont tournés dans un repère quasi inertiel par l’angle sidéral
            de Greenwich, la vitesse étant corrigée de la rotation terrestre. On en déduit les
            éléments orbitaux osculateurs : demi-grand axe, excentricité, inclinaison et
            argument de latitude.
          </p>
          <p>
            Le demi-grand axe osculateur est inutilisable tel quel : l’aplatissement de la Terre
            (terme J2) le fait osciller d’environ ±9 km deux fois par révolution, alors qu’une
            manœuvre de routine le déplace de quelques mètres. Le terme court-période du
            premier ordre, a<sub>osc</sub> − a<sub>moy</sub> = 3/2 · J2 · R² / a · sin² i ·
            cos 2u, est retiré analytiquement à chaque instant, puis la valeur est moyennée sur
            chaque révolution, d’un passage au nœud ascendant au suivant. Retirer ce terme avant
            de moyenner rend aussi la moyenne insensible à la position des bornes de la
            révolution sur la grille de 10 s.
          </p>
        </Section>

        <Section id="trace" title="Signature de la trace au sol">
          <p>
            Sentinel-1 suit une orbite à répétition : sa trace au sol revient exactement sur
            elle-même toutes les {model?.repeat_revolutions ?? 175} révolutions (12 jours). Les
            irrégularités du champ de gravité impriment donc aux éléments moyens une signature
            qui dépend uniquement de la position sur la trace et se répète à l’identique à
            chaque cycle. Ce qui ressemblait à du bruit dans les premières versions du
            détecteur était en réalité, pour l’essentiel, cette signature.
          </p>
          <p>
            Un gabarit de {model?.repeat_revolutions ?? 175} valeurs est estimé sur les
            périodes calmes de {period.cal} uniquement, en alternant deux étapes : retirer une
            tendance lente de chaque segment calme, puis moyenner les résidus par position dans
            le cycle. Son amplitude atteint
            {model ? ` ${length(model.template_ptp_a_m)} crête à crête sur le demi-grand axe et ${num(model.template_ptp_i_mdeg, 1)} millidegrés sur l’inclinaison` : " plusieurs dizaines de mètres"}.
            {m && <> Une fois retiré, le bruit d’une révolution à l’autre (révolutions calmes
            de la période de calibration) passe de {length(m.noise_m.raw)} à{" "}
            {length(m.noise_m.template)} sur le demi-grand axe</>}
            {" "}et le gain se maintient sur {period.test}, jamais vu pendant l’estimation. La même
            correction appliquée à l’inclinaison rend visibles les manœuvres hors plan.
          </p>
        </Section>

        <Section id="trainee" title="Modèle de traînée">
          <p>
            Entre deux manœuvres, le demi-grand axe corrigé décroît presque linéairement, d’autant
            plus vite que le Soleil est actif. La pente de chaque segment calme de la période de
            calibration est ajustée par une loi de puissance de l’activité solaire :
          </p>
          <p className="formula">
            −da/drév = e<sup>{beta ? num(beta[0], 2) : "b₀"}</sup> · F81<sup>{beta ? num(beta[1], 2) : "b₁"}</sup> ·
            (F/F81)<sup>{beta ? num(beta[2], 2) : "b₂"}</sup> · (1 + Ap)<sup>{beta ? num(beta[3], 2) : "b₃"}</sup>
          </p>
          <p>
            où F est le flux F10.7 du jour et F81 sa moyenne sur les 81 jours précédents. Une
            loi exponentielle ajustée sur les mêmes segments prédit
            {model?.drag_2024_ratio
              ? ` ${num(model.drag_2024_ratio.exponential, 2)} fois la décroissance observée au maximum solaire de 2024, contre ${num(model.drag_2024_ratio.power_law, 2)} pour la loi de puissance`
              : " une décroissance bien trop forte au maximum solaire de 2024"}
            ; 2024 est hors de la plage de calibration, et c’est la loi de puissance qui est
            retenue. Constat honnête : ce modèle explique bien la
            décroissance mais n’améliore pas la détection, une droite ajustée localement faisant
            aussi bien ; il reste utile pour interpréter les séries et comme prédicteur physique.
          </p>
        </Section>

        <Section id="detection" title="Détection des manœuvres">
          <p>
            <strong>Détecteur de référence.</strong> Pour chaque révolution, on compare la moyenne
            du demi-grand axe sur les W révolutions suivantes à celle des W précédentes, en
            sautant la révolution de la poussée, partiellement décalée. La statistique est
            normalisée de façon robuste (médiane, écart absolu médian, sur une fenêtre centrée
            d’une quinzaine de jours, donc non causale) et une alarme est levée au-delà d’un
            seuil. Une variante, normalisée à partir des seules différences entre points
            successifs, tourne dans le labo.
          </p>
          <p>
            <strong>Détecteur principal (CUSUM).</strong> Sur le demi-grand axe corrigé de la
            signature et de la traînée, et sur l’inclinaison corrigée, la valeur attendue à chaque
            révolution est prédite à partir des
            {m ? ` ${m.detector.memory}` : ""} révolutions précédentes du segment en cours.
            L’écart standardisé alimente une somme cumulée bilatérale (test de Page), qui
            n’accumule que les écarts supérieurs à κ
            {m ? ` = ${num(Number(m.detector.kappa), 1)}` : ""} écarts-types et déclenche au-delà
            de h{m ? ` = ${num(Number(m.detector.h), 0)}` : ""}. Les écarts sont écrêtés, pour
            qu’une révolution aberrante isolée ne suffise pas, et l’échelle du bruit est estimée
            sur les révolutions récentes, ce qui absorbe les périodes où la traînée est moins
            bien modélisée. La décision à une révolution donnée n’utilise que les révolutions
            passées ; la correction de traînée utilise le F10.7 et l’Ap du jour (au plus un
            jour d’avance sur ces indices) et une moyenne de F10.7 glissante vers le passé.
            Les délais sont comptés en temps orbital : ils n’incluent pas le délai de
            publication des orbites précises (environ trois semaines). L’instant de rupture est
            la dernière révolution où la somme était nulle.
          </p>
          <p>
            <strong>Estimation du Δv.</strong> Le saut de demi-grand axe est mesuré par deux droites
            ajustées de part et d’autre de la rupture, puis converti en Δv le long de la trace
            par l’équation de Gauss pour une orbite quasi circulaire : Δv ≈ Δa · v / (2a).
          </p>
        </Section>

        <Section id="classification" title="Classification des événements">
          <p>
            Chaque détection est rangée dans l’une de trois classes : maintien à poste,
            changement d’orbite (inclinaison, séquence, abaissement) ou détection inexpliquée.
            L’historique ESA n’ayant pas de classe « anomalie », cette dernière classe est
            définie opérationnellement comme une détection sans manœuvre ESA correspondante :
            manœuvre absente de l’historique, artefact de
            restitution d’orbite ou, le plus souvent, effet de la traînée lors d’un orage
            géomagnétique.
          </p>
          <p>
            Deux approches ont été comparées sur {period.test} : une règle sur les sauts estimés
            (saut d’inclinaison supérieur à 5 σ ou demi-grand axe en baisse : changement
            d’orbite ; hausse significative : maintien à poste ; sinon : inexpliquée) et un
            modèle de gradient boosting entraîné sur {period.cal}. Sur {period.test}, les deux
            obtiennent des scores très proches (le modèle légèrement devant en F1 macro, la
            règle en exactitude) : la période d’entraînement (Soleil calme) contient trop peu de
            détections inexpliquées pour que l’apprentissage fasse nettement mieux.
            L’application affiche la règle, plus simple et lisible. Ce choix a été fait au vu
            des scores de test ; ce n’est donc pas une sélection hors échantillon.
          </p>
        </Section>

        <Section id="evaluation" title="Évaluation">
          <p>
            La vérité terrain est l’historique ESA. Une détection est correcte si elle tombe à
            une révolution près d’une manœuvre ; plusieurs alarmes sur une même manœuvre ne
            comptent qu’une fois et ne sont pas des fausses alarmes. Tous les détecteurs sont
            évalués sur les mêmes manœuvres (même fenêtre d’observabilité
            {m?.observe_window ? ` de ${m.observe_window} révolutions` : ""}). Tous les réglages sont
            choisis sur{model ? ` ${dateFr(model.operational_start)} – ${dateFr(model.split)}` : " la période de calibration"} et
            les résultats rapportés sur la période suivante, jusqu’à la fin de l’historique ESA :
            aucune information du futur ne sert à régler le passé. La phase d’acquisition de
            l’orbite de référence
            {model ? ` (avant le ${dateFr(model.operational_start)})` : ""} est exclue.
          </p>
          {m && (
            <Reveal variant="fade" className="table-scroll">
            <table className="metrics">
              <thead>
                <tr><th>Détecteur (période de test)</th><th>Rappel</th><th>Précision</th><th>F1</th>
                    <th>Erreur Δv</th><th>Délai (orbite)</th></tr>
              </thead>
              <tbody>
                {m.comparison.map((row) => (
                  <tr key={String(row.detector)}>
                    <td>{detectorName(String(row.detector))}</td>
                    <td>{pct(Number(row["test recall"]))}</td>
                    <td>{pct(Number(row["test precision"]))}</td>
                    <td>{num(Number(row["test f1"]), 2)}</td>
                    <td>{dv(Number(row["test Δv err [mm/s]"]))}</td>
                    <td>{num(Number(row["test delay [h]"]), 1)} h{row["causal delay"] ? "" : " *"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </Reveal>
          )}
          <p>
            * Délai indicatif : la normalisation des détecteurs de référence utilise une fenêtre
            centrée, ils ne peuvent donc pas lever l’alarme aussi tôt en conditions réelles.
            Le détecteur principal gagne surtout en rappel sur les petites manœuvres et sur les
            manœuvres d’inclinaison ; la référence corrigée reste plus précise. Les fausses
            alarmes du détecteur principal se concentrent au maximum solaire, en particulier
            autour des orages géomagnétiques.
          </p>
        </Section>

        <Section id="degradation" title="Dégradation et laboratoire">
          <p>
            Pour mesurer la robustesse, les éphémérides précises sont dégradées : erreur de
            position σ sur chaque axe et erreur de vitesse proportionnelle (celle d’une erreur
            d’orbite de même amplitude), corrélation temporelle des erreurs ρ entre points
            successifs, et sous-échantillonnage à N points par jour. La chaîne de traitement est
            ensuite rejouée : éléments, moyenne par révolution (ou par groupe de révolutions
            quand les points sont rares), correction de la signature, détecteur de référence.
          </p>
          <p>
            La carte du labo est une grille précalculée hors ligne : pour chaque combinaison de
            {g ? ` ${g.axes.sigma_m.length} niveaux de bruit, ${g.axes.dv_cm_s.length} tailles de manœuvre, ${g.axes.points_per_day.length} cadences et ${g.axes.rho.length} corrélations` : " bruit, taille de manœuvre, cadence et corrélation"},
            une manœuvre de Δv connu est injectée dans des fenêtres calmes réelles
            {g ? ` (${g.trials_per_cell} essais par case)` : ""}, et la probabilité de la retrouver
            est mesurée. Plus petite manœuvre détectée à 90 % selon le préréglage :
            {g && <> POD Copernicus {presetFloor(g, "pod")}, Radar {presetFloor(g, "radar")},
            Type TLE {presetFloor(g, "tle")}</>}. Les manœuvres de routine de Sentinel-1 restent
            donc invisibles avec des données de type catalogue public.
          </p>
          <p>
            Dans le navigateur, un Web Worker recalcule la série à chaque mouvement de curseur à
            partir d’états réels à une minute d’intervalle ; il reproduit l’implémentation
            Python à 10⁻⁵ m près, ce que vérifient des tests de parité sur des jeux de données
            partagés à graine fixe, rejoués des deux côtés. L’échelle du détecteur y est estimée à partir des différences
            entre points successifs, qu’une manœuvre ne perturbe qu’une fois, ce qui le rend
            robuste aux fenêtres contenant plusieurs manœuvres.
          </p>
        </Section>

        <Section id="limites" title="Limites connues">
          <ul>
            <li>Le préréglage « Type TLE » est une approximation : un vrai TLE est un jeu
              d’éléments moyens SGP4 dont les erreurs, surtout le long de la trace, ne sont pas
              gaussiennes.</li>
            <li>Les réglages ont été calibrés en période de Soleil calme ; au maximum solaire de
              2024, les orages géomagnétiques produisent davantage de fausses alarmes, et un
              seuil plus élevé aurait mieux fonctionné a posteriori.</li>
            <li>Les types de manœuvre sont déduits de la géométrie des poussées, pas fournis
              par l’ESA.</li>
            <li>Les deux anomalies connues ont été réexaminées sur la série corrigée : l’impact
              de particule sur Sentinel-1A (août 2016) ne laisse aucun saut au-delà des
              fluctuations des révolutions calmes ; pour la panne d’alimentation de Sentinel-1B
              (décembre 2021), rien non plus sur la partie de la journée testable, le reste
              étant masqué par une manœuvre la veille au soir (rapport « anomalies » du
              dépôt).</li>
            <li>Avec un échantillonnage clairsemé, des termes à courte période non modélisés
              fixent un plancher de détection indépendant du bruit ; un gabarit à l’échelle de
              l’échantillon l’abaisserait, mais supposerait des connaissances issues d’orbites
              précises.</li>
            <li>Le labo du navigateur travaille à un point par minute (10 s hors ligne).</li>
            <li>Les délais de détection sont en temps orbital et n’incluent pas le délai de
              publication des orbites précises (environ trois semaines).</li>
          </ul>
        </Section>

        <Section id="gouvernance" title="Data governance">
          <p>
            Satwiser n’utilise que des données ouvertes et redistribuables : orbites précises et
            historiques de manœuvres Copernicus Sentinel-1 (ESA), indices F10.7 et Ap du GFZ
            Potsdam (CC BY 4.0, la colonne de nombre de taches solaires, sous licence non
            commerciale, étant écartée). Aucune donnée Space-Track n’est utilisée : son accord
            d’utilisation interdit de transférer données et analyses dérivées à des tiers, et la
            diffusion de ces informations relève du 10 U.S.C. § 2274. Les fichiers bruts ne sont
            pas stockés dans le dépôt : un script de collecte les retélécharge, et seuls des
            agrégats sont servis par l’application. Contient des données Copernicus Sentinel
            modifiées (2014–2026).
          </p>
        </Section>

        <Section id="suite" title="Et ensuite">
          <p>
            Détecter une manœuvre → mettre à jour l’orbite → réévaluer le risque de collision.
            Le troisième maillon fait l’objet d’un projet séparé,{" "}
            <a href={FCD_URL} target="_blank" rel="noreferrer">False Calm Detector</a>. Le code,
            les rapports de chaque étape et les scripts qui produisent tous les chiffres de cette
            page sont sur{" "}
            <a href={REPO_URL} target="_blank" rel="noreferrer">github.com/victorvb22/satwiser</a>.
          </p>
        </Section>
      </div>
    </>
  );
}
