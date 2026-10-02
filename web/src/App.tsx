import { useEffect, useState } from "react";
import Runs from "./screens/Runs";
import Compare from "./screens/Compare";
import Divergence from "./screens/Divergence";
import Study from "./screens/Study";

const ROUTES = ["study", "runs", "compare", "divergence"] as const;
type Route = (typeof ROUTES)[number];

function routeFromHash(): Route {
  const r = window.location.hash.replace(/^#\/?/, "").split("?")[0];
  return (ROUTES as readonly string[]).includes(r) ? (r as Route) : "study";
}

export default function App() {
  const [route, setRoute] = useState<Route>(routeFromHash());
  useEffect(() => {
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <nav>
        <span className="brand">flipbook</span>
        {ROUTES.map((r) => (
          <a key={r} href={`#/${r}`} className={r === route ? "active" : ""}>
            {r}
          </a>
        ))}
      </nav>
      <main>
        {route === "runs" && <Runs />}
        {route === "compare" && <Compare />}
        {route === "divergence" && <Divergence />}
        {route === "study" && <Study />}
      </main>
    </div>
  );
}
