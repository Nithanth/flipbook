import { useEffect, useState } from "react";
import { api, type DivergenceRow } from "../api";
import Tip from "../Tip";

interface Pair {
  base: string;
  ckpt: string;
}

export default function Divergence() {
  const [pairs, setPairs] = useState<Pair[]>([]);
  const q = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
  const initial =
    q.get("base") && q.get("ckpt") ? `${q.get("base")}__${q.get("ckpt")}` : "";
  const [sel, setSel] = useState(initial);
  const [rows, setRows] = useState<DivergenceRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.divergencePairs().then((ps) => {
      setPairs(ps);
      if (ps.length && !sel) setSel(`${ps[0].base}__${ps[0].ckpt}`);
    });
  }, []);

  useEffect(() => {
    if (!sel) return;
    const [base, ckpt] = sel.split("__");
    setRows(null);
    setErr(null);
    api.divergence(base, ckpt).then(setRows).catch((e) => setErr(String(e)));
  }, [sel]);

  return (
    <>
      <h1>divergence</h1>
      {pairs.length === 0 ? (
        <p className="sub">
          no divergence data yet — run <code>flipbook diverge --base … --ckpt …</code>
        </p>
      ) : (
        <div className="row">
          <label>
            pair{" "}
            <select value={sel} onChange={(e) => setSel(e.target.value)}>
              {pairs.map((p) => (
                <option key={`${p.base}__${p.ckpt}`} value={`${p.base}__${p.ckpt}`}>
                  {p.base.slice(0, 12)} → {p.ckpt.slice(0, 12)}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      {err && <p className="err">{err}</p>}
      {rows && (
        <table>
          <thead>
            <tr>
              <th>row</th>
              <th><Tip text="Total log-prob shift of the checkpoint vs base, summed over every token of the baseline's own trace. Negative = the checkpoint finds this trace less likely.">Σ nats</Tip></th>
              <th><Tip text="Token position where the two models' per-token log-probs first diverge materially — roughly where the checkpoint starts reasoning differently.">first div</Tip></th>
              <th><Tip text="Probability of ending the response at token 0 (emit nothing), base → checkpoint. A jump means the checkpoint wants to skip the question entirely.">p_skip</Tip></th>
              <th><Tip text="Per-token log-prob delta along the baseline trace, compressed to a sparkline. Red bars = checkpoint less confident than base there; the amber tick marks first divergence.">trace</Tip></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.row_id}:${r.sample_idx}`}>
                <td className="mono">{r.row_id}:{r.sample_idx}</td>
                <td className={r.sum_nats < 0 ? "neg" : "pos"}>
                  {r.sum_nats.toFixed(1)}
                </td>
                <td>{r.divergence_pos ?? "—"}</td>
                <td>
                  {r.p_skip_base.toFixed(3)} → {r.p_skip_ckpt.toFixed(3)}
                </td>
                <td>
                  <Spark delta={r.delta} mark={r.divergence_pos} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

/** Delta sparkline: red = ckpt less confident than base, blue = more. */
function Spark({ delta, mark }: { delta: number[]; mark: number | null }) {
  const w = 220;
  const h = 28;
  if (!delta.length) return null;
  const step = Math.max(1, Math.floor(delta.length / w));
  const pts: number[] = [];
  for (let i = 0; i < delta.length; i += step) pts.push(delta[i]);
  const max = Math.max(1e-9, ...pts.map(Math.abs));
  const bw = w / pts.length;
  return (
    <svg width={w} height={h} className="spark">
      <line x1={0} x2={w} y1={h / 2} y2={h / 2} style={{ stroke: "var(--border)" }} />
      {pts.map((d, i) => {
        const bh = (Math.abs(d) / max) * (h / 2);
        return (
          <rect
            key={i}
            x={i * bw}
            y={d < 0 ? h / 2 : h / 2 - bh}
            width={Math.max(1, bw - 0.4)}
            height={bh}
            style={{ fill: d < 0 ? "var(--neg)" : "var(--accent)" }}
          />
        );
      })}
      {mark != null && (
        <line
          x1={(mark / delta.length) * w}
          x2={(mark / delta.length) * w}
          y1={0}
          y2={h}
          style={{ stroke: "var(--warn)" }}
          strokeWidth={2}
        />
      )}
    </svg>
  );
}
