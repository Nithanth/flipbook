import { useEffect, useState } from "react";
import { api, type DivergencePair as Pair, type DivergenceRow, type Run } from "../api";
import Tip from "../Tip";
import { BranchView, Spark, TraceView } from "../Trace";
import Md from "../Md";
import { runLabel } from "../runLabel";

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

  const label = (id: string) => runLabel(runs.find((x) => x.run_id === id), id);

  useEffect(() => {
    if (!sel) return;
    const [base, ckpt] = sel.split("__");
    setRows(null);
    setErr(null);
    api.divergence(base, ckpt).then(setRows).catch((e) => setErr(String(e)));
  }, [sel]);

  // deep link: ?row=…&sample=… opens and scrolls to that sample's row
  useEffect(() => {
    if (!rows) return;
    const row = q.get("row");
    const sample = q.get("sample");
    if (row == null || sample == null) return;
    const key = `${row}:${sample}`;
    if (!rows.some((r) => `${r.row_id}:${r.sample_idx}` === key)) return;
    setOpen(key);
    document.getElementById(`drow-${key}`)?.scrollIntoView();
  }, [rows]);

  return (
    <>
      <h1>divergence</h1>
      <p className="page-sub">where in the baseline's own reasoning do the two models stop agreeing?</p>
      {pairs.length === 0 ? (
        <p className="sub">
          no divergence data yet - run <code>flipbook diverge --base … --ckpt …</code>
        </p>
      ) : (
        <div className="row">
          <label>
            pair{" "}
            <select value={sel} onChange={(e) => setSel(e.target.value)}>
              {pairs.map((p) => (
                <option key={`${p.base}__${p.ckpt}`} value={`${p.base}__${p.ckpt}`}>
                  {label(p.base)} → {label(p.ckpt)}
                  {p.self ? " (noise floor)" : ""}
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
        <PairRead
          rows={rows}
          self={sel.split("__")[0] === sel.split("__")[1]}
          floor={pairs.find((p) => `${p.base}__${p.ckpt}` === sel)?.noise_floor ?? null}
        />
      )}
      {rows && (() => {
        // when ~every row departs at token 0-2 the column is dead - fold it into the read
        const early = rows.filter((r) => r.divergence_pos != null);
        const collapseDiv =
          early.length >= rows.length * 0.8 && early.every((r) => r.divergence_pos! <= 2);
        return (
        <table>
          <thead>
            <tr>
              <th>row</th>
              <th><Tip text="Total log-prob shift of the checkpoint vs base, summed over every token of the baseline's own trace. Negative = the checkpoint finds this trace less likely.">Σ nats</Tip></th>
              {!collapseDiv && (
                <th><Tip text="Token position where the two models' per-token log-probs first diverge materially - roughly where the checkpoint starts reasoning differently.">first div</Tip></th>
              )}
              <th><Tip text="Probability of ending the response at token 0 (emit nothing), base → checkpoint. A jump means the checkpoint wants to skip the question entirely.">p_skip</Tip></th>
              <th><Tip text="Per-token log-prob delta along the baseline trace, compressed to a sparkline. Red bars = checkpoint less confident than base there; the amber tick marks first divergence.">trace</Tip></th>
            </tr>
          </thead>
          <tbody>
            {[...rows]
              .sort((a, b) => a.sum_nats - b.sum_nats)
              .map((r) => {
              const short = r.row_id.includes(":")
                ? r.row_id.split(":").slice(-1)[0]
                : r.row_id;
              const key = `${r.row_id}:${r.sample_idx}`;
              const isOpen = open === key;
              return [
                <tr
                  key={key}
                  id={`drow-${key}`}
                  className="fliprow"
                  onClick={() => setOpen(isOpen ? null : key)}
                >
                  <td className="mono">
                    <Tip text={r.q ? `${short}:${r.sample_idx} - ${r.q}` : r.row_id}>
                      <a
                        href={`#/compare?a=${sel.split("__")[0]}&b=${sel.split("__")[1]}&row=${encodeURIComponent(r.row_id)}`}
                        onClick={(e) => e.stopPropagation()}
                        title="open this question in compare"
                      >{`${short}:${r.sample_idx}`}</a>
                    </Tip>
                    {r.q && <div className="sub qcell"><Md text={r.q} /></div>}
                  </td>
                  <td className={r.sum_nats < 0 ? "neg" : "pos"}>
                    {r.sum_nats.toFixed(1)}
                  </td>
                  {!collapseDiv && <td>{r.divergence_pos ?? "-"}</td>}
                  <td>
                    {r.p_skip_base.toFixed(3)} → {r.p_skip_ckpt.toFixed(3)}
                  </td>
                  <td>
                    <Spark delta={r.delta} mark={r.divergence_pos} n={r.n ?? undefined} />
                  </td>
                </tr>,
                isOpen && (
                  <ExpandedRow
                    key={`${key}-x`}
                    colSpan={collapseDiv ? 4 : 5}
                    base={sel.split("__")[0]}
                    ckpt={sel.split("__")[1]}
                    r={r}
                  />
                ),
              ];
            })}
          </tbody>
        </table>
        );
      })()}
    </>
  );
}

/** Pair-level "the read": what the deltas collectively say before you open a row. */
function PairRead({
  rows,
  self,
  floor,
}: {
  rows: DivergenceRow[];
  self: boolean;
  floor: number | null;
}) {
  if (!rows.length) return null;
  const n = rows.length;
  const med = (xs: number[]) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const medSum = med(rows.map((r) => r.sum_nats));
  const fracs = rows
    .filter((r) => r.divergence_pos != null && r.n)
    .map((r) => r.divergence_pos! / r.n!);
  const meanSkipB = rows.reduce((a, r) => a + r.p_skip_base, 0) / n;
  const meanSkipC = rows.reduce((a, r) => a + r.p_skip_ckpt, 0) / n;

  const bullets: string[] = self
    ? [
        "this is the same checkpoint scored twice - the instrument's A/A noise floor. Σ nats near zero and faint traces are expected; anything here is scoring noise, not a policy shift",
      ]
    : [
        `median Σ ${medSum.toFixed(1)} nats across ${n} rows - ${
          Math.abs(medSum) < 5
            ? "the checkpoint still finds these baseline traces about as likely as base did"
            : "the checkpoint assigns these baseline traces substantially different probability"
        }`,
      ];
  if (!self && floor != null && Math.abs(medSum) > floor * 3) {
    bullets.push(
      `instrument floor |Σ| ≤ ${floor.toFixed(1)} nats (same checkpoint rescored) - this pair is far above scoring noise`,
    );
  }
  const early = rows.filter((r) => r.divergence_pos != null);
  if (early.length >= n * 0.8 && early.every((r) => r.divergence_pos! <= 2)) {
    bullets.push(
      `departure begins at token 0-2 on ${early.length}/${n} rows - the checkpoint disagrees from the first generated token, not mid-reasoning`,
    );
  } else if (fracs.length >= 2) {
    const f = med(fracs);
    bullets.push(
      `divergence typically begins ${(f * 100).toFixed(0)}% into the trace - the checkpoint ${
        f > 0.6 ? "follows the baseline's reasoning before breaking late" : "departs from baseline reasoning early"
      }`,
    );
  }
  bullets.push(
    `mean p_skip ${meanSkipB.toFixed(3)} → ${meanSkipC.toFixed(3)} - ${
      meanSkipC > meanSkipB * 3 && meanSkipC > 0.05
        ? "the checkpoint increasingly wants to emit nothing"
        : "no collapse toward empty responses"
    }`,
  );

  return (
    <div className="card read">
      <div className="mini-label">the read</div>
      {bullets.map((b) => (
        <p key={b}>{b}</p>
      ))}
    </div>
  );
}

/** Expanded row: the trace + the branch probe, sharing a "cut" position that
 *  clicks on a token set. */
function ExpandedRow({
  colSpan,
  base,
  ckpt,
  r,
}: {
  colSpan: number;
  base: string;
  ckpt: string;
  r: DivergenceRow;
}) {
  const [cut, setCut] = useState<number | null>(r.divergence_pos ?? r.win_argmin);
  return (
    <tr>
      <td colSpan={colSpan} className="tracexp">
        <TraceView
          base={base}
          ckpt={ckpt}
          row={r.row_id}
          sample={r.sample_idx}
          delta={r.delta}
          mark={r.divergence_pos}
          onPick={setCut}
          cut={cut}
        />
        <div className="sub">
          baseline sample {r.sample_idx} · {(r.n ?? r.delta.length).toLocaleString()} tokens ·
          Σ {r.sum_nats.toFixed(1)} nats · mean {r.mean_nats.toFixed(4)} nats/token
          {r.divergence_pos != null
            ? ` · first divergence at token ${r.divergence_pos.toLocaleString()}`
            : ""}
        </div>
        <BranchView
          base={base}
          ckpt={ckpt}
          row={r.row_id}
          sample={r.sample_idx}
          pos={cut}
          nTokens={r.n ?? r.delta.length}
        />
      </td>
    </tr>
  );
}
