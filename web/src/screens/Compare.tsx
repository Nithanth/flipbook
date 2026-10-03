import { useEffect, useState } from "react";
import { api, type PairReport, type Run } from "../api";
import Tip from "../Tip";

function hashParams(): URLSearchParams {
  return new URLSearchParams(window.location.hash.split("?")[1] ?? "");
}

export default function Compare() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [a, setA] = useState(hashParams().get("a") ?? "");
  const [b, setB] = useState(hashParams().get("b") ?? "");
  const [pair, setPair] = useState<PairReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.runs().then((rs) => {
      setRuns(rs);
      if (rs.length >= 2 && (!a || !b)) {
        // default: baseline vs latest checkpoint on the most-populated manifest
        const step = (r: Run) =>
          (r.provenance?.train_step_measured as number) ??
          r.train_step ??
          (r.label === "base" || r.label === "baseline" ? 0 : 1e9);
        const byManifest = new Map<string, Run[]>();
        for (const r of rs) {
          const k = r.manifest_hash ?? "";
          byManifest.set(k, [...(byManifest.get(k) ?? []), r]);
        }
        const group = [...byManifest.values()].sort((x, y) => y.length - x.length)[0];
        const sorted = group.sort((x, y) => step(x) - step(y));
        if (!a) setA(sorted[0].run_id);
        if (!b) setB(sorted[sorted.length - 1].run_id);
      }
    });
  }, []);

  useEffect(() => {
    if (!a || !b || a === b) return;
    setPair(null);
    setErr(null);
    api.compare(a, b).then(setPair).catch((e) => setErr(String(e)));
  }, [a, b]);

  return (
    <>
      <h1>compare</h1>
      <div className="row">
        <RunPicker label="base" runs={runs} value={a} onChange={setA} />
        <RunPicker label="ckpt" runs={runs} value={b} onChange={setB} />
      </div>
      {pair && (
        <ContextLine pair={pair} runs={runs} />
      )}
      {err && <p className="err">{err}</p>}
      {pair && <Report pair={pair} runs={runs} />}
    </>
  );
}

/** What am I comparing? study · manifest · label names, for someone who landed via a deep link. */
function ContextLine({ pair, runs }: { pair: PairReport; runs: Run[] }) {
  const ra = runs.find((r) => r.run_id === pair.run_a);
  const rb = runs.find((r) => r.run_id === pair.run_b);
  const bits = [
    ra?.study ?? rb?.study,
    ra?.manifest ?? ra?.manifest_hash?.slice(0, 10),
    ra?.model ?? rb?.model,
  ].filter(Boolean);
  return (
    <p className="sub" style={{ marginTop: -8 }}>
      {ra?.label ?? pair.run_a.slice(0, 8)} → {rb?.label ?? pair.run_b.slice(0, 8)}
      {bits.length ? ` · ${bits.join(" · ")}` : ""}
    </p>
  );
}

function RunPicker({
  label,
  runs,
  value,
  onChange,
}: {
  label: string;
  runs: Run[];
  value: string;
  onChange: (v: string) => void;
}) {
  // group by study so cross-study pairs are a deliberate choice, not the default mess
  const studies = [...new Set(runs.map((r) => r.study ?? "ungrouped"))].sort();
  return (
    <label>
      {label}{" "}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {studies.map((s) => (
          <optgroup key={s} label={s}>
            {runs
              .filter((r) => (r.study ?? "ungrouped") === s)
              .sort((x, y) => (x.train_step ?? -1) - (y.train_step ?? -1))
              .map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.label ?? r.run_id.slice(0, 12)} · {r.run_id.slice(0, 8)}
                </option>
              ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

function Report({ pair, runs }: { pair: PairReport; runs: Run[] }) {
  const ag = pair.agreement;
  const labelA = runs.find((r) => r.run_id === pair.run_a)?.label ?? "base";
  const labelB = runs.find((r) => r.run_id === pair.run_b)?.label ?? "ckpt";
  const failKinds = [
    ...new Set([...Object.keys(pair.failures.a), ...Object.keys(pair.failures.b)]),
  ].sort();
  return (
    <>
      <div className="cards">
        <div className="card">
          <div className="big">
            {pct(pair.acc_a)} → {pct(pair.acc_b)}
          </div>
          <div className="sub">
            <Tip text="Paired per-question accuracy change (b − a), with a bootstrap 95% CI over the shared eval questions. Pairing removes question-difficulty noise; a CI that crosses 0 is inconclusive.">
              Δ {pair.delta >= 0 ? "+" : ""}
              {pair.delta.toFixed(3)} [{pair.delta_ci[0].toFixed(3)},{" "}
              {pair.delta_ci[1].toFixed(3)}]
            </Tip>
            {" · "}
            {pair.n_pairs} paired rows
          </div>
        </div>
        <div className="card">
          <div className="big">
            {pair.flips.length} flips
          </div>
          <div className="sub">
            {pair.flips.filter((f) => f.kind === "regression").length} regressions ·{" "}
            {pair.flips.filter((f) => f.kind === "gain").length} gains
          </div>
        </div>
        <div className="card">
          <div className="big">
            {pct(pair.truncation_rate_a)} → {pct(pair.truncation_rate_b)}
          </div>
          <div className="sub">truncation rate</div>
        </div>
        <div className="card">
          <div className="big">
            <Tip text="Average response length in generated tokens. A big paired shift means the output regime changed — rambling or collapsing — not just accuracy.">
              {pair.tokens.a.mean.toFixed(0)} → {pair.tokens.b.mean.toFixed(0)}
            </Tip>
          </div>
          <div className="sub">
            gen tokens · Δ {pair.delta_tokens >= 0 ? "+" : ""}
            {pair.delta_tokens.toFixed(0)} [{pair.delta_tokens_ci[0].toFixed(0)},{" "}
            {pair.delta_tokens_ci[1].toFixed(0)}]
          </div>
        </div>
        <div className="card">
          <div className="big">
            ${pair.cost_a.toFixed(2)} vs ${pair.cost_b.toFixed(2)}
          </div>
          <div className="sub">estimated cost</div>
        </div>
      </div>

      <h2>agreement</h2>
      <table className="agreement">
        <tbody>
          <tr>
            <td>both right <b>{ag.both_right}</b></td>
            <td>only {labelA} right <b>{ag.a_only}</b></td>
          </tr>
          <tr>
            <td>only {labelB} right <b>{ag.b_only}</b></td>
            <td>both wrong <b>{ag.both_wrong}</b></td>
          </tr>
        </tbody>
      </table>

      {failKinds.length > 0 && (
        <>
          <h2>
            <Tip text="Why the wrong samples were wrong: truncated at the token cap, no parseable answer, wrong answer, etc. This is the shape of the failure — the difference between 'degenerate outputs' and 'just wrong'.">
              failure kinds
            </Tip>
          </h2>
          <table>
            <thead>
              <tr>
                <th>kind</th>
                <th>{labelA}</th>
                <th>{labelB}</th>
                <th>Δ</th>
              </tr>
            </thead>
            <tbody>
              {failKinds.map((k) => {
                const na = pair.failures.a[k] ?? 0;
                const nb = pair.failures.b[k] ?? 0;
                return (
                  <tr key={k}>
                    <td>{k.replace(/_/g, " ")}</td>
                    <td>{na}</td>
                    <td>{nb}</td>
                    <td className={nb - na > 0 ? "neg" : nb - na < 0 ? "pos" : ""}>
                      {nb - na >= 0 ? "+" : ""}
                      {nb - na}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      {pair.flips.length > 0 && (
        <>
          <h2>flips</h2>
          <table>
            <thead>
              <tr>
                <th>row</th>
                <th><Tip text="Base model's pass rate on this question across its k samples.">p(base)</Tip></th>
                <th><Tip text="Checkpoint's pass rate on this question across its k samples.">p(ckpt)</Tip></th>
                <th><Tip text="regression = right→wrong, gain = wrong→right. (hard) = every sample flipped, not just a marginal one — the strongest evidence of a real change.">kind</Tip></th>
              </tr>
            </thead>
            <tbody>
              {pair.flips.map((f) => (
                <tr key={f.row_id} className={f.kind}>
                  <td className="mono">
                    <a href={`#/divergence?base=${pair.run_a}&ckpt=${pair.run_b}`}>
                      {f.row_id}
                    </a>
                  </td>
                  <td>{f.p_a.toFixed(2)}</td>
                  <td>{f.p_b.toFixed(2)}</td>
                  <td>
                    {f.kind}
                    {f.hard ? " (hard)" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {pair.excluded.length > 0 && (
        <>
          <h2>excluded ({pair.excluded.length})</h2>
          <table>
            <tbody>
              {pair.excluded.map((e) => (
                <tr key={e.row_id}>
                  <td className="mono">{e.row_id}</td>
                  <td>{e.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </>
  );
}

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}
