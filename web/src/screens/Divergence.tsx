import { useEffect, useState } from "react";
import { api, type DivergenceRow, type Run } from "../api";
import Tip from "../Tip";

interface Pair {
  base: string;
  ckpt: string;
}

export default function Divergence() {
  const [pairs, setPairs] = useState<Pair[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const q = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
  const initial =
    q.get("base") && q.get("ckpt") ? `${q.get("base")}__${q.get("ckpt")}` : "";
  const [sel, setSel] = useState(initial);
  const [rows, setRows] = useState<DivergenceRow[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.runs().then(setRuns);
    api.divergencePairs().then((ps) => {
      setPairs(ps);
      if (ps.length && !sel) setSel(`${ps[0].base}__${ps[0].ckpt}`);
    });
  }, []);

  // hash → human label; fall back to step tag, then a short hash
  const label = (id: string) => {
    const r = runs.find((x) => x.run_id === id);
    return r?.label ?? (r?.train_step != null ? `step${r.train_step}` : id.slice(0, 10));
  };

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
                  {label(p.base)} → {label(p.ckpt)}
                  {p.base === p.ckpt ? " (noise floor)" : ""}
                </option>
              ))}
            </select>
          </label>
          <a className="sub" href={`#/compare?a=${sel.split("__")[0]}&b=${sel.split("__")[1]}`}>
            open in compare →
          </a>
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
            {rows.map((r) => {
              const key = `${r.row_id}:${r.sample_idx}`;
              const isOpen = open === key;
              return [
                <tr
                  key={key}
                  className="fliprow"
                  onClick={() => setOpen(isOpen ? null : key)}
                >
                  <td className="mono">{key}</td>
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
                </tr>,
                isOpen && (
                  <tr key={`${key}-x`}>
                    <td colSpan={5} className="tracexp">
                      <Spark delta={r.delta} mark={r.divergence_pos} w={800} h={72} />
                      <div className="sub">
                        {r.delta.length.toLocaleString()} tokens · Σ {r.sum_nats.toFixed(1)} nats ·
                        mean {r.mean_nats.toFixed(4)} nats/token
                        {r.divergence_pos != null
                          ? ` · first divergence at token ${r.divergence_pos.toLocaleString()}`
                          : ""}
                      </div>
                    </td>
                  </tr>
                ),
              ];
            })}
          </tbody>
        </table>
      )}
    </>
  );
}

/** Delta sparkline: red = ckpt less confident than base, blue = more. */
function Spark({
  delta,
  mark,
  w = 220,
  h = 28,
}: {
  delta: number[];
  mark: number | null;
  w?: number;
  h?: number;
}) {
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
