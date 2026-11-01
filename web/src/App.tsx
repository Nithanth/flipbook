import { useEffect, useState } from "react";
import Runs from "./screens/Runs";
import Compare from "./screens/Compare";
import Divergence from "./screens/Divergence";
import Study from "./screens/Study";

const ROUTES = ["study", "evals", "compare", "divergence"] as const;
type Route = (typeof ROUTES)[number];
type Theme = "dark" | "light";

function routeFromHash(): Route {
  const r = window.location.hash.replace(/^#\/?/, "").split("?")[0];
  return (ROUTES as readonly string[]).includes(r) ? (r as Route) : "study";
}

function initialTheme(): Theme {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/** Three cards fanning from a shared pivot - a page mid-flip, same as favicon. */
function Logo() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <g transform="rotate(32 2.5 21)">
        <rect x="2.5" y="4.5" width="11.5" height="16.5" rx="2" stroke="#5e7cff" strokeOpacity={0.35} strokeWidth="1.8" />
      </g>
      <g transform="rotate(16 2.5 21)">
        <rect x="2.5" y="4.5" width="11.5" height="16.5" rx="2" stroke="#5e7cff" strokeOpacity={0.65} strokeWidth="1.8" />
      </g>
      <rect x="2.5" y="4.5" width="11.5" height="16.5" rx="2" fill="#5e7cff" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2M12 19.5v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2.5 12h2M19.5 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20.5 13.5A8.5 8.5 0 1 1 10.5 3.5a7 7 0 0 0 10 10z" />
    </svg>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash());
  const [theme, setTheme] = useState<Theme>(initialTheme());

  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <div className="app">
      <nav>
        <a className="brand" href="#/study"><Logo />flipbook</a>
        {ROUTES.map((r) => (
          <a key={r} href={`#/${r}`} className={r === route ? "active" : ""}>
            {r}
          </a>
        ))}
        <span className="spacer" />
        <button
          className="theme-btn"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "switch to light" : "switch to dark"}
          aria-label={theme === "dark" ? "switch to light" : "switch to dark"}
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </nav>
      <main>
        {route === "evals" && <Runs />}
        {route === "compare" && <Compare />}
        {route === "divergence" && <Divergence />}
        {route === "study" && <Study />}
      </main>
    </div>
  );
}
