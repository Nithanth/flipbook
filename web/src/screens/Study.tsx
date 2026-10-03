import { useEffect, useMemo, useState } from "react";
import { api, type Run, type StudyDetail } from "../api";
import Tip from "../Tip";

/** Study dashboard: the whole story of one training run on one page. */
export default function Study() {
  const [studies, setStudies] = useState<string[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [study, setStudy] = useState(
    new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("name") ?? "",
  );
  const [detail, setDetail] = useState<StudyDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.studies(), api.runs()]).then(([ss, rs]) => {
      setStudies(ss);
      setRuns(rs);
    });
  }, []);

  // default: the study with the most runs — that's the story, not the leftovers
  useEffect(() => {
    if (study || !studies.length) return;
    const counts = new Map<string, number>();
    for (const r of runs) if (r.study) counts.set(r.study, (counts.get(r.study) ?? 0) + 1);
    const best = [...studies].sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0))[0];
    setStudy(best);
  }, [studies, runs, study]);

  useEffect(() => {
    if (!study) return;
    setDetail(null);
    setErr(null);
    api.study(study).then(setDetail).catch((e) => setErr(String(e)));
  }, [study]);

  const derived = useMemo(() => (detail ? derive(detail) : null), [detail]);

  return (
    <>
      <h1>study</h1>
      <div className="row">
        <label>
          study{" "}
          <select value={study} onChange={(e) => setStudy(e.target.value)}>
            {studies.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        {detail && <span className="sub">{detail.runs.length} runs · {derived?.metricKeys.length ?? 0} metrics</span>}
      </div>
      {err && <p className="err">{err}</p>}
      {detail && derived && (
        <>
          <Hero d={derived} runs={detail.runs} />
          <MetricStrip d={derived} />
          <StepTable d={derived} runs={detail.runs} />
        </>
      )}
    </>
  );
}

interface StepRow {
  step: number;
  pass1?: number;
  delta?: number;
  ci?: [number, number];
  regressions?: number;
  gains?: number;
  trunc?: number;
  genTok?: number;
  div?: number;
  pSkip?: number;
  cost?: number;
  loss?: number;
  effortGap?: number;
}

interface Derived {
  rows: StepRow[];
  metricKeys: string[];
  lossSeries: { step: number; value: number }[];
}

function derive(detail: StudyDetail): Derived {
  const m = detail.metrics;
  const pick = (suffix: string) => {
    const k = Object.keys(m).find((x) => x.endsWith(suffix));
    return k ? new Map(m[k].map((r) => [r.step, r.value])) : new Map<number, number>();
  };
  const pass1 = pick("/pass1");
  const delta = pick("/delta_vs_base");
  const ciLo = pick("/delta_ci_lo");
  const ciHi = pick("/delta_ci_hi");
  const reg = pick("/regressions");
  const gains = pick("/gains");
  const trunc = pick("/truncation_rate");
  const genTok = pick("/mean_gen_tokens");
  const div = pick("/divergence_mean_nats");
  const pSkip = pick("/p_skip");
  const cost = pick("/cost_usd");
  const lossSeries = m["train_mean_nll"] ?? m["train_mean_bpb"] ?? [];

  // per-step effort gap comes from the parquet rollups keyed by run_id
  const stepByRun = new Map<number, string>();
  for (const r of detail.runs) if (r.train_step != null) stepByRun.set(r.train_step, r.run_id);
  const gapByStep = new Map<number, number>();
  for (const [rid, g] of Object.entries(detail.effort_gap)) {
    const st = detail.runs.find((r) => r.run_id === rid)?.train_step;
    if (st != null) gapByStep.set(st, g);
  }

  const steps = [...new Set([...pass1.keys(), ...lossSeries.map((s) => s.step)])].sort(
    (a, b) => a - b,
  );
  const rows: StepRow[] = steps.map((step) => ({
    step,
    pass1: pass1.get(step),
    delta: delta.get(step),
    ci:
      ciLo.get(step) != null && ciHi.get(step) != null
        ? [ciLo.get(step)!, ciHi.get(step)!]
        : undefined,
    regressions: reg.get(step),
    gains: gains.get(step),
    trunc: trunc.get(step),
    genTok: genTok.get(step),
    div: div.get(step),
    pSkip: pSkip.get(step),
    cost: cost.get(step),
    loss: lossSeries.find((s) => s.step === step)?.value,
    effortGap: gapByStep.get(step),
  }));
  return { rows, metricKeys: Object.keys(m), lossSeries };
}

/** pass1 (left axis) vs train loss (right axis) — the collision chart. */
function Hero({ d, runs }: { d: Derived; runs: Run[] }) {
  const pts = d.rows.filter((r) => r.pass1 != null);
  const lossMax = Math.max(1e-9, ...d.lossSeries.map((s) => s.value));
  if (!pts.length) return <p className="sub">no eval metrics for this study</p>;

  const w = 860;
  const h = 280;
  const pad = 44;
  const x0 = Math.min(...pts.map((r) => r.step));
  const x1 = Math.max(...pts.map((r) => r.step));
  const sx = (x: number) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (w - 2 * pad);
  const syPass = (y: number) => h - pad - Math.max(0, Math.min(1, y)) * (h - 2 * pad);
  const syLoss = (y: number) => h - pad - (y / lossMax) * (h - 2 * pad);

  const passPath = pts.map((r, i) => `${i ? "L" : "M"}${sx(r.step)},${syPass(r.pass1!)}`).join(" ");
  const lossPath = d.lossSeries
    .map((r, i) => `${i ? "L" : "M"}${sx(r.step)},${syLoss(r.value)}`)
    .join(" ");

  const runFor = (step: number) => runs.find((r) => r.train_step === step);
  const base = runs.find((r) => r.label === "baseline" || r.label === "base" || r.train_step == null);
  const go = (step: number) => {
    const r = runFor(step);
    if (base && r && r.run_id !== base.run_id)
      window.location.hash = `#/compare?a=${base.run_id}&b=${r.run_id}`;
  };

  return (
    <div className="hero">
      <svg width={w} height={h} className="chart">
        <line x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        <line x1={pad} x2={pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        <line x1={w - pad} x2={w - pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        {lossPath && (
          <path d={lossPath} fill="none" style={{ stroke: "var(--accent2)" }} strokeWidth={1.4} strokeDasharray="4 3" />
        )}
        <path d={passPath} fill="none" style={{ stroke: "var(--accent)" }} strokeWidth={2} />
        {pts.map((r) => (
          <circle
            key={r.step}
            cx={sx(r.step)}
            cy={syPass(r.pass1!)}
            r={5}
            style={{ fill: "var(--accent)", stroke: "var(--bg)", cursor: "pointer" }}
            strokeWidth={1.5}
            onClick={() => go(r.step)}
          >
            <title>{`step ${r.step}: pass1 ${(r.pass1! * 100).toFixed(1)}% — click to compare`}</title>
          </circle>
        ))}
        <text x={6} y={pad} style={{ fill: "var(--accent)" }} fontSize={11}>100%</text>
        <text x={6} y={h - pad} style={{ fill: "var(--accent)" }} fontSize={11}>0%</text>
        <text x={w - pad + 6} y={pad} style={{ fill: "var(--accent2)" }} fontSize={11}>{lossMax.toPrecision(2)}</text>
        <text x={pad} y={h - 12} style={{ fill: "var(--muted)" }} fontSize={11}>{x0}</text>
        <text x={w - pad - 20} y={h - 12} style={{ fill: "var(--muted)" }} fontSize={11}>{x1}</text>
      </svg>
      <div className="legend">
        <Tip text="Fraction of frozen eval questions answered correctly at this checkpoint (left axis). Click a point to compare that step against baseline.">
          <span style={{ color: "var(--accent)" }}>━ pass1 (click a point)</span>
        </Tip>
        <Tip text="Training loss on the fine-tuning batches — what the optimizer sees (right axis). It can keep falling while eval behavior collapses; that's the collision this chart exists to show.">
          <span style={{ color: "var(--accent2)" }}>┅ train_mean_nll</span>
        </Tip>
      </div>
    </div>
  );
}

function MetricStrip({ d }: { d: Derived }) {
  const cards: [string, string, (r: StepRow) => number | undefined, (v: number) => string][] = [
    ["divergence (nats)", "How far the checkpoint's token probabilities moved from base on the baseline's own reasoning traces. ~0 = unchanged policy; very negative = the internals shifted hard — often before accuracy shows it.", (r) => r.div, (v) => v.toFixed(0)],
    ["mean gen tokens", "Average response length. A sudden drop or spike signals a degenerate output regime (rambling into the cap, or collapsing to short format-locked answers).", (r) => r.genTok, (v) => v.toFixed(0)],
    ["truncation", "Fraction of samples that hit the max-token cap before finishing. High truncation = the model rambles and never emits a final answer.", (r) => r.trunc, (v) => `${(v * 100).toFixed(0)}%`],
    ["effort gap (nats)", "Log-prob difference between effort=0.9 and effort=0.2 prompts on the same trace. Large = effort conditioning still modulates the model; ~0 = the dial is dead.", (r) => r.effortGap, (v) => v.toFixed(0)],
    ["p_skip", "The model's own probability of ending the response immediately (end-of-message as the first token). Rising p_skip = it increasingly wants to emit nothing.", (r) => r.pSkip, (v) => v.toPrecision(2)],
    ["eval cost", "Sampling cost of this eval point, in USD.", (r) => r.cost, (v) => `$${v.toFixed(2)}`],
  ];
  return (
    <div className="strip">
      {cards.map(([label, tip, get, fmt]) => (
        <div className="mini" key={label}>
          <div className="mini-label"><Tip text={tip}>{label}</Tip></div>
          <TinySeries
            pts={d.rows.filter((r) => get(r) != null).map((r) => ({ x: r.step, y: get(r)! }))}
          />
          <div className="mini-val">
            {(() => {
              const vs = d.rows.filter((r) => get(r) != null);
              return vs.length ? fmt(get(vs[vs.length - 1])!) : "—";
            })()}
          </div>
        </div>
      ))}
    </div>
  );
}

function TinySeries({ pts }: { pts: { x: number; y: number }[] }) {
  const w = 150;
  const h = 40;
  if (pts.length < 2) return <svg width={w} height={h} />;
  const [x0, x1] = [Math.min(...pts.map((p) => p.x)), Math.max(...pts.map((p) => p.x))];
  const [y0, y1] = [Math.min(...pts.map((p) => p.y)), Math.max(...pts.map((p) => p.y))];
  const sx = (x: number) => 2 + ((x - x0) / Math.max(1, x1 - x0)) * (w - 4);
  const sy = (y: number) => h - 2 - ((y - y0) / Math.max(1e-9, y1 - y0)) * (h - 4);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ");
  return (
    <svg width={w} height={h}>
      <path d={path} fill="none" style={{ stroke: "var(--muted)" }} strokeWidth={1.4} />
      {pts.map((p) => (
        <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={2} style={{ fill: "var(--muted)" }}>
          <title>{`step ${p.x}: ${p.y}`}</title>
        </circle>
      ))}
    </svg>
  );
}

function StepTable({ d, runs }: { d: Derived; runs: Run[] }) {
  const base = runs.find((r) => r.label === "baseline" || r.label === "base" || r.train_step == null);
  const runFor = (step: number) => runs.find((r) => r.train_step === step);
  const evalRows = d.rows.filter((r) => r.pass1 != null);
  if (!evalRows.length) return null;
  return (
    <>
      <h2>per-step evals</h2>
      <table>
        <thead>
          <tr>
            <th>step</th>
            <th><Tip text="Accuracy on the frozen eval manifest at this checkpoint.">pass1</Tip></th>
            <th><Tip text="Paired per-question accuracy change vs baseline, with a bootstrap 95% CI over questions. Deltas inside the noise band (~±0.07 at n=30, k=2) are inconclusive.">Δ vs base</Tip></th>
            <th><Tip text="Questions that changed correctness vs baseline. R = right→wrong regressions, G = wrong→right gains. Noise flips are roughly symmetric; one-directional flips signal a real shift.">flips</Tip></th>
            <th><Tip text="Fraction of samples truncated at the max-token cap.">trunc</Tip></th>
            <th><Tip text="Average response length in tokens.">gen tok</Tip></th>
            <th><Tip text="Mean per-token log-prob shift of the checkpoint vs base, scored on the baseline's own traces. ~0 = unchanged; very negative = large internal shift.">div nats</Tip></th>
            <th><Tip text="Effort-conditioning gap in nats. Large = the effort dial still works; ~0 = conditioning collapsed.">effort gap</Tip></th>
            <th><Tip text="Estimated sampling cost of this eval point." right>cost</Tip></th>
          </tr>
        </thead>
        <tbody>
          {evalRows.map((r) => {
            const run = runFor(r.step);
            const link =
              base && run && run.run_id !== base.run_id
                ? `#/compare?a=${base.run_id}&b=${run.run_id}`
                : undefined;
            return (
              <tr key={r.step}>
                <td>
                  {link ? <a href={link}>step {r.step}</a> : `step ${r.step}`}
                </td>
                <td>{r.pass1 != null ? `${(r.pass1 * 100).toFixed(1)}%` : "—"}</td>
                <td className={r.delta != null && r.delta < 0 ? "neg" : "pos"}>
                  {r.delta != null
                    ? `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(2)} [${r.ci?.[0].toFixed(2)}, ${r.ci?.[1].toFixed(2)}]`
                    : "—"}
                </td>
                <td>
                  {r.regressions != null ? `${r.regressions}R/${r.gains}G` : "—"}
                </td>
                <td>{r.trunc != null ? `${(r.trunc * 100).toFixed(0)}%` : "—"}</td>
                <td>{r.genTok != null ? r.genTok.toFixed(0) : "—"}</td>
                <td className={r.div != null && r.div < -50 ? "neg" : ""}>
                  {r.div != null ? r.div.toFixed(0) : "—"}
                </td>
                <td>{r.effortGap != null ? r.effortGap.toFixed(0) : "—"}</td>
                <td>{r.cost != null ? `$${r.cost.toFixed(2)}` : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}
