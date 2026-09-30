import { useEffect, useMemo, useState, type ReactNode } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { mulberry32 } from "../lab/pipeline";

export const ATTRIBUTION = "Contains modified Copernicus Sentinel data 2014–2026";
export const REPO_URL = "https://github.com/victorvb22/satwiser";
export const FCD_URL = "https://github.com/victorvb22/false-calm-detector";

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <ellipse cx="14" cy="14" rx="12" ry="5" stroke="#9CC2FF" strokeWidth="1"
               transform="rotate(-24 14 14)" />
      <circle cx="14" cy="14" r="3" fill="#E8ECF4" />
      <circle cx="24" cy="9.5" r="1.6" fill="#F4A259" />
    </svg>
  );
}

/** Static, very faint star field (deterministic). */
function Stars() {
  const stars = useMemo(() => {
    const r = mulberry32(3);
    return Array.from({ length: 110 }, () => ({
      x: r() * 100, y: r() * 100, s: r() < 0.85 ? 1 : 2, o: 0.12 + r() * 0.45,
    }));
  }, []);
  return (
    <div className="stars" aria-hidden="true">
      {stars.map((st, k) => (
        <span key={k} style={{ position: "absolute", left: `${st.x}%`, top: `${st.y}%`,
                               width: st.s, height: st.s, borderRadius: "50%",
                               background: "#CFDBF5", opacity: st.o }} />
      ))}
    </div>
  );
}

export function Layout() {
  return (
    <>
      <Stars />
      <div className="shell">
        <nav className="nav" aria-label="Navigation principale">
          <NavLink to="/" className="brand" aria-label="Satwiser, accueil">
            <Logo />
            <span>Satwiser</span>
          </NavLink>
          <div className="nav-links">
            <NavLink to="/" end>Mission</NavLink>
            <NavLink to="/labo">Labo</NavLink>
            <NavLink to="/methode">Méthode</NavLink>
          </div>
          <span className="badge">DONNÉES COPERNICUS · POD</span>
        </nav>
        <main>
          <Outlet />
        </main>
        <footer className="footer">
          <span>{ATTRIBUTION}</span>
          <span>
            <a href={REPO_URL} target="_blank" rel="noreferrer">Code source</a>
          </span>
        </footer>
      </div>
    </>
  );
}

/** Loading state that explains the hosted API's cold start after a few seconds. */
export function Loading({ what }: { what: string }) {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setSlow(true), 3500);
    return () => clearTimeout(timer);
  }, []);
  return (
    <div className="loading" role="status" aria-live="polite">
      <div className="pulse" />
      <span>Chargement {what}…</span>
      {slow && (
        <span className="faint">
          Le serveur gratuit se réveille après une période d’inactivité (jusqu’à une minute).
        </span>
      )}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return <div className="loading" role="alert">{children}</div>;
}
