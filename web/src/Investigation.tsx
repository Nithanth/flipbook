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
  return <section className="card read" aria-label="Regression investigation">
    <h2>Regression investigation</h2>
    <p>Mean score {percent(pair.acc_a)} → {percent(pair.acc_b)} across {pair.n_pairs} paired questions.
      {" "}{pair.agreement.a_only} regressed, {pair.agreement.b_only} improved, and {pair.agreement.both_right + pair.agreement.both_wrong} kept the same majority outcome.</p>
    {pair.excluded.length > 0 && <p>{pair.excluded.length} questions excluded from the paired comparison.</p>}
    {leading ? <>
      <p>Checkpoint failure labels on regressed questions (a question can have several labels):</p>
      <ul>{kinds.map(k => <li key={k.kind}>{k.kind}: {k.rows.length} of {drops.length} questions</li>)}</ul>
    </> : drops.length > 0 && <p>No failure labels were recorded for the regressed questions.</p>}
    {first && <h3>Answers to inspect</h3>}
    {[first, second].map(c => {
      if (!c) return null;
      const trace = traces.find(t => t.row_id === c.row_id && t.divergence_pos != null)
        ?? traces.find(t => t.row_id === c.row_id);
      const query = trace && new URLSearchParams({ base: pair.run_a, ckpt: pair.run_b, row: c.row_id, sample: String(trace.sample_idx) });
      return <div key={c.row_id} className="investigation-example">
        <button onClick={() => inspect(c.row_id)}>{c.q || c.row_id}</button>
        <div className="sub">Score {percent(c.p_a!)} → {percent(c.p_b!)}
          {leading && (outside === c ? ` · Exception to the “${leading.kind}” pattern` :
            leading.rows.includes(c) ? ` · Example of “${leading.kind}”` : "")}
          {query && trace && <> · <a href={`#/divergence?${query}`}>Inspect baseline trace{trace.divergence_pos != null ? ` at token ${trace.divergence_pos}` : ""}</a></>}
        </div>
      </div>;
    })}
    {traceError && <p className="sub">{traceError}</p>}
  </section>;
}
