import { useEffect, useState } from "react";
import { api, type DivergenceRow, type PairReport } from "./api";

export default function Investigation({ pair, inspect }: {
  pair: PairReport;
  inspect: (row: string) => void;
}) {
  const [traces, setTraces] = useState<DivergenceRow[]>([]);
  const [traceError, setTraceError] = useState("");
  useEffect(() => {
    if (pair.comparability?.token_views === false || !pair.n_pairs) return;
    let active = true;
    api.divergence(pair.run_a, pair.run_b).then(rows => {
      if (active) setTraces(rows);
    }).catch(e => {
      if (active && !String(e).includes("404")) setTraceError("Recorded traces could not be loaded.");
    });
    return () => { active = false; };
  }, [pair]);

  if (!pair.n_pairs) return <p>No complete paired questions to investigate.</p>;
  const cells = (pair.cells ?? []).filter(c => c.p_a != null && c.p_b != null);
  const drops = cells.filter(c => c.cell === "a_only");
  const kinds = Object.entries(pair.failure_rows_b ?? {}).map(([kind, ids]) => {
    const recorded = new Set(ids);
    const labels: Record<string, string> = { empty: "empty response", wrong_answer: "wrong answer", truncation: "token limit reached" };
    return { kind: labels[kind] ?? kind.replaceAll("_", " "), rows: drops.filter(c => recorded.has(c.row_id)) };
  }).filter(k => k.rows.length).sort((a, b) => b.rows.length - a.rows.length || a.kind.localeCompare(b.kind));
  const leading = kinds[0];
  const byDrop = (a: typeof cells[number], b: typeof cells[number]) =>
    (b.p_a! - b.p_b!) - (a.p_a! - a.p_b!) || a.row_id.localeCompare(b.row_id);
  const candidates = (drops.length ? drops : cells).slice().sort(byDrop);
  const first = (leading ? leading.rows.slice().sort(byDrop) : candidates)[0];
  const outside = leading && candidates.find(c => !leading.rows.includes(c));
  const second = outside ?? candidates.find(c => c !== first);
  const percent = (x: number) => (100 * x).toFixed(1) + "%";
  return <section className="investigation-layout" aria-label="Regression investigation">
    <div className="investigation-patterns">
      <h2>Failure patterns</h2>
      {leading ? <>
        <p className="sub">Questions affected out of {drops.length} regressions · labels can overlap</p>
        {kinds.map(k => <div className="failure-bar" key={k.kind}>
          <span>{k.kind}</span><strong>{k.rows.length} / {drops.length}</strong>
          <div><i style={{width: `${100 * k.rows.length / drops.length}%`}} /></div>
        </div>)}
      </> : <p>{drops.length ? "No recorded failure labels." : "No majority outcomes regressed."}</p>}
      {pair.excluded.length > 0 && <p className="sub">{pair.excluded.length} excluded questions</p>}
    </div>
    <div className="investigation-evidence">
    {first && <h2>Answers to inspect</h2>}
    {[first, second].map(c => {
      if (!c) return null;
      const trace = traces.find(t => t.row_id === c.row_id && t.divergence_pos != null)
        ?? traces.find(t => t.row_id === c.row_id);
      const query = trace && new URLSearchParams({ base: pair.run_a, ckpt: pair.run_b, row: c.row_id, sample: String(trace.sample_idx) });
      return <div key={c.row_id} className="investigation-example">
        {leading && <span className="sub">{outside === c ? "Exception" : "Example"} · {leading.kind}</span>}
        <button onClick={() => inspect(c.row_id)}>{c.q || c.row_id}</button>
        <div className="sub">Score {percent(c.p_a!)} → {percent(c.p_b!)}
          {query && trace && <> · <a href={`#/divergence?${query}`}>Token trace →</a></>}
        </div>
      </div>;
    })}
    {traceError && <p className="sub">{traceError}</p>}
    </div>
  </section>;
}
