import { useEffect, useMemo, useState, type MouseEvent } from "react";
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

  // default: the study with the most runs - that's the story, not the leftovers
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
  const [focus, setFocus] = useState<string | null>(null);

  const base = detail?.runs.find(
    (r) => r.label === "baseline" || r.label === "base" || r.train_step == null,
  );
  const go = (step: number) => {
    const r = detail?.runs.find((x) => x.train_step === step);
    if (base && r && r.run_id !== base.run_id)
      window.location.hash = `#/compare?a=${base.run_id}&b=${r.run_id}`;
  };

  const models = [...new Set(detail?.runs.map((r) => r.model).filter(Boolean))];
  const nEvals = derived?.rows.filter((r) => r.pass1 != null).length ?? 0;

  return (
    <>
      <h1>study</h1>
      <p className="page-sub">one training run, every checkpoint: did the eval move, and what moved first?</p>
      <div className="row">
        <label>
          study{" "}
          <select value={study} onChange={(e) => setStudy(e.target.value)}>
            {studies.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        {detail && (
          <span className="sub">
            {detail.runs.length} runs · {nEvals} eval points
            {models.length === 1 ? ` · ${models[0]}` : ""}
          </span>
        )}
      </div>
      {err && <p className="err">{err}</p>}
      {detail && derived && derived.rows.length === 0 && (
        <div className="card read">
          <div className="mini-label">no training metrics</div>
          <p>
            this study has {detail.runs.length} eval run{detail.runs.length === 1 ? "" : "s"} but no
            tracked training loop - there's no step axis to chart. pair runs up in{" "}
            <a href="#/compare">compare</a> or browse them under <a href="#/evals">evals</a>.
          </p>
        </div>
      )}
      {detail && derived && derived.rows.length > 0 && (
        <>
          <Hero d={derived} go={go} />
          <Narrative d={derived} />
          <MetricStrip d={derived} focus={focus} onFocus={setFocus} go={go} />
          <StepTable d={derived} runs={detail.runs} />
        </>
      )}
    </>
  );
}

interface StepRow {
  step: number;
  pass1?: number;
  passK?: number;
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

/** Round x ticks: smallest 1/2/5/8-ish step that yields <=6 ticks. Shared by
 *  the hero and focus charts so they line up. */
function tickSteps(x0: number, x1: number): number[] {
  const span = Math.max(1, x1 - x0);
  const step =
    [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 500, 640, 1000]
      .find((t) => span / t <= 6) ?? Math.ceil(span / 6);
  const out: number[] = [];
  for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) out.push(x);
  return out;
}

interface Derived {
  rows: StepRow[];
  metricKeys: string[];
  lossKey: string | null;
  lossSeries: { step: number; value: number }[];
  epochStarts: number[];
}

function derive(detail: StudyDetail): Derived {
  const m = detail.metrics;
  const pick = (suffix: string) => {
    const k = Object.keys(m).find((x) => x.endsWith(suffix));
    return k ? new Map(m[k].map((r) => [r.step, r.value])) : new Map<number, number>();
  };
  const pass1 = pick("/pass1");
  // the evaluator emits pass{n} for n in 2..k; keep the largest n
  const passKKey = Object.keys(m)
    .map((x) => ({ x, n: Number(/\/pass(\d+)$/.exec(x)?.[1]) }))
    .filter((k) => k.n > 1)
    .sort((a, b) => b.n - a.n)[0]?.x;
  const passK = passKKey
    ? new Map(m[passKKey].map((r) => [r.step, r.value]))
    : new Map<number, number>();
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
  // next-token nll for SFT runs; bpb is the same quantity in bits.
  // other recipes (RL) log different keys and just show no loss line
  const lossKey = ["train_mean_nll", "train_mean_bpb"].find((k) => m[k]?.length) ?? null;
  const lossSeries = lossKey ? m[lossKey] : [];

  // epoch boundaries: steps where the epoch counter ticks up
  const epochKey = Object.keys(m).find((x) => x === "epoch" || x.endsWith("/epoch"));
  const epochStarts: number[] = [];
  if (epochKey) {
    const pts = [...m[epochKey]].sort((x, y) => x.step - y.step);
    for (let i = 1; i < pts.length; i++) {
      if (pts[i].value > pts[i - 1].value) epochStarts.push(pts[i].step);
    }
  }

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
    passK: passK.get(step),
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
  return { rows, metricKeys: Object.keys(m), lossKey, lossSeries, epochStarts };
}

/** pass1 (left axis) vs train loss (right axis) - the collision chart. */
function Hero({ d, go }: { d: Derived; go: (step: number) => void }) {
  const [hover, setHover] = useState<number | null>(null);
  const pts = d.rows.filter((r) => r.pass1 != null);
  const lossMax = Math.max(1e-9, ...d.lossSeries.map((s) => s.value));
  if (!pts.length) return <p className="sub">no eval metrics for this study</p>;

  const w = 860;
  const h = 280;
  const pad = 44;
  const allSteps = [...pts.map((r) => r.step), ...d.lossSeries.map((s) => s.step)];
  const x0 = Math.min(...allSteps);
  const x1 = Math.max(...allSteps);
  const sx = (x: number) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (w - 2 * pad);
  const xticks = tickSteps(x0, x1);
  const syPass = (y: number) => h - pad - Math.max(0, Math.min(1, y)) * (h - 2 * pad);
  // 1-indexed data pass containing this step (boundaries are where the
  // logged epoch counter ticks up)
  const epochAt = (s: number) => 1 + d.epochStarts.filter((e) => e <= s).length;
  const syLoss = (y: number) => h - pad - (y / lossMax) * (h - 2 * pad);

  const passPath = pts.map((r, i) => `${i ? "L" : "M"}${sx(r.step)},${syPass(r.pass1!)}`).join(" ");
  const lossPath = d.lossSeries
    .map((r, i) => `${i ? "L" : "M"}${sx(r.step)},${syLoss(r.value)}`)
    .join(" ");

  // snap to every logged step - evals are sparse, loss lands every step
  const hoverSteps = [
    ...new Set([...pts.map((r) => r.step), ...d.lossSeries.map((s) => s.step)]),
  ].sort((a, b) => a - b);
  const hovRow = hover != null ? pts.find((r) => r.step === hover) : undefined;
  const hovLoss = hover != null ? d.lossSeries.find((s) => s.step === hover)?.value : undefined;
  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * w;
    let best = hoverSteps[0];
    let bd = Infinity;
    for (const s of hoverSteps) {
      const dd = Math.abs(sx(s) - mx);
      if (dd < bd) {
        bd = dd;
        best = s;
      }
    }
    setHover(best);
  };

  // modal gap between eval steps, for the legend's "evals run every N steps"
  const gaps = pts.slice(1).map((r, i) => r.step - pts[i].step);
  const evalGap = gaps.length
    ? [...gaps].sort(
        (x, y) =>
          gaps.filter((g) => g === y).length - gaps.filter((g) => g === x).length,
      )[0]
    : null;

  return (
    <div className="hero">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        width="100%"
        height="auto"
        preserveAspectRatio="xMidYMid meet"
        className="chart"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <line x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        <line x1={pad} x2={pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        <line x1={w - pad} x2={w - pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        {/* mid gridline at 50% pass + x ticks */}
        <line x1={pad} x2={w - pad} y1={syPass(0.5)} y2={syPass(0.5)} style={{ stroke: "var(--border)" }} strokeDasharray="2 5" strokeOpacity={0.7} />
        {xticks.map((x) => (
          <g key={x}>
            <line x1={sx(x)} x2={sx(x)} y1={h - pad} y2={h - pad + 4} style={{ stroke: "var(--muted)" }} strokeOpacity={0.6} />
            <text x={sx(x)} y={h - pad + 16} textAnchor="middle" fontSize={10} style={{ fill: "var(--muted)" }}>{x}</text>
          </g>
        ))}
        {d.epochStarts.map((s, i) => (
          <g key={s}>
            <line
              x1={sx(s)}
              x2={sx(s)}
              y1={pad}
              y2={h - pad}
              style={{ stroke: "var(--border)" }}
              strokeDasharray="2 4"
            />
            {/* fat invisible sibling so a 1px dashed line is hoverable */}
            <line x1={sx(s)} x2={sx(s)} y1={pad} y2={h - pad} strokeWidth={10} stroke="transparent">
              <title>{`epoch ${i + 2} begins (step ${s}) - the optimizer starts another pass over the training set`}</title>
            </line>
          </g>
        ))}
        {lossPath && (
          <path d={lossPath} fill="none" style={{ stroke: "var(--accent2)" }} strokeWidth={1.4} strokeDasharray="4 3" />
        )}
        <path d={passPath} fill="none" style={{ stroke: "var(--accent)" }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
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
            <title>{`step ${r.step} (epoch ${epochAt(r.step)}): pass1 ${(r.pass1! * 100).toFixed(1)}% - click to compare`}</title>
          </circle>
        ))}
        {hover != null && (
          <g pointerEvents="none" className="hovmark">
            <line
              x1={sx(hover)}
              x2={sx(hover)}
              y1={pad}
              y2={h - pad}
              style={{ stroke: "var(--border)" }}
              strokeDasharray="3 3"
            />
            {hovRow ? (
              <circle
                cx={sx(hover)}
                cy={syPass(hovRow.pass1!)}
                r={6.5}
                fill="none"
                style={{ stroke: "var(--accent)" }}
                strokeWidth={1.5}
              />
            ) : (
              hovLoss != null && (
                <circle
                  cx={sx(hover)}
                  cy={syLoss(hovLoss)}
                  r={3.5}
                  style={{ fill: "var(--accent2)" }}
                />
              )
            )}
            <ChartTip
              x={sx(hover) > w - 190 ? sx(hover) - 176 : sx(hover) + 14}
              y={pad + 6}
              lines={[
                [`step ${hover} · epoch ${epochAt(hover)}`, "var(--text)", true],
                ...(hovRow
                  ? ([[`pass@1 ${(hovRow.pass1! * 100).toFixed(1)}%`, "var(--accent)", false]] as [string, string, boolean][])
                  : []),
                ...(hovLoss != null
                  ? ([[`nll  ${hovLoss.toPrecision(3)}`, "var(--accent2)", false]] as [string, string, boolean][])
                  : []),
              ]}
            />
          </g>
        )}
        <text x={6} y={pad} style={{ fill: "var(--accent)" }} fontSize={11}>100%</text>
        <text x={6} y={h - pad} style={{ fill: "var(--accent)" }} fontSize={11}>0%</text>
        <text x={w - pad + 6} y={pad} style={{ fill: "var(--accent2)" }} fontSize={11}>{lossMax.toPrecision(2)}</text>
        <text x={w / 2} y={h - 4} textAnchor="middle" style={{ fill: "var(--muted)" }} fontSize={11}>
          optimizer step →
        </text>
        <text x={w - pad + 6} y={h - pad} style={{ fill: "var(--accent2)" }} fontSize={11}>0</text>
      </svg>
      <div className="legend">
        <Tip text="Fraction of frozen eval questions answered correctly at this checkpoint (left axis). Click a point to compare that step against baseline.">
          <span style={{ color: "var(--accent)" }}>━ pass@1 (click a point)</span>
        </Tip>
        {d.lossKey && (
          <Tip text={`${d.lossKey}: training loss on the fine-tuning batches - what the optimizer sees (right axis). Logged every optimizer step (one batch); evals run every ${evalGap ?? "?"} steps. It can keep falling while eval behavior collapses; that's the collision this chart exists to show.`}>
            <span style={{ color: "var(--accent2)" }}>┅ {d.lossKey}</span>
          </Tip>
        )}
      </div>
      <p className="sub" style={{ marginTop: 4 }}>
        x = optimizer step (one training batch).
        {d.lossSeries.length ? " train loss is logged every step; " : " "}
        each pass@1 dot is a full eval sweep
        {evalGap ? `, run every ${evalGap} steps` : ""}.
      </p>
    </div>
  );
}

/** In-SVG hover label: lines = [text, cssColor, bold]. */
function ChartTip({
  x,
  y,
  lines,
}: {
  x: number;
  y: number;
  lines: [string, string, boolean][];
}) {
  const lh = 15;
  const wBox = 162;
  const hBox = lines.length * lh + 14;
  return (
    <g transform={`translate(${x},${y})`}>
      <rect
        width={wBox}
        height={hBox}
        rx={7}
        style={{ fill: "var(--surface-2)", stroke: "var(--border)" }}
      />
      {lines.map(([t, c, b], i) => (
        <text
          key={i}
          x={10}
          y={14 + i * lh}
          fontSize={11}
          fontWeight={b ? 650 : 400}
          style={{ fill: c, fontVariantNumeric: "tabular-nums" }}
        >
          {t}
        </text>
      ))}
    </g>
  );
}

/** Plain-English takeaway, derived from the step rows - the chart shows it, this says it. */
function Narrative({ d }: { d: Derived }) {
  const evals = d.rows.filter((r) => r.pass1 != null);
  if (evals.length < 2) return null;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const bullets: string[] = [];

  const peak = evals.reduce((a, b) => (b.pass1! > a.pass1! ? b : a));
  const after = evals.filter((r) => r.step >= peak.step);
  const trough = after.reduce((a, b) => (b.pass1! < a.pass1! ? b : a));
  const last = evals[evals.length - 1];
  const first = evals[0];
  const collapsed = peak.pass1! - trough.pass1! > 0.05 && last.pass1! < peak.pass1! - 0.05;
  const improved = last.pass1! - first.pass1! > 0.05;
  const verdict = collapsed
    ? "the fine-tune regressed this eval"
    : improved
      ? "the fine-tune improved this eval"
      : peak.pass1! - trough.pass1! > 0.05
        ? "pass@1 dipped mid-training and recovered"
        : "no eval-visible change";
  // did divergence move before the first significant accuracy drop?
  const cliff = evals.find((r) => r.ci && r.ci[1] < 0);
  if (peak.pass1! - trough.pass1! > 0.05) {
    // a "peak" inside the CI of the first point is noise - anchor on where it held
    const realPeak = peak.step !== first.step && peak.ci && peak.ci[0] > 0;
    const held = cliff ? evals[Math.max(0, evals.indexOf(cliff) - 1)] : peak;
    const open = realPeak
      ? `pass@1 rose to ${pct(peak.pass1!)} (step ${peak.step})`
      : `pass@1 held near ${pct(first.pass1!)} through step ${held.step}`;
    bullets.push(
      `${open}, then fell to ${pct(trough.pass1!)} by step ${trough.step}` +
        (last.pass1! > trough.pass1! + 0.02 ? ` - ending at ${pct(last.pass1!)} (step ${last.step})` : ` and stayed there`),
    );
  } else {
    bullets.push(
      `pass@1 stayed flat through training (${pct(evals[0].pass1!)} → ${pct(last.pass1!)}) - no eval-visible regression`,
    );
  }

  const divMove = evals.find((r) => r.div != null && r.div < -50);
  if (cliff && divMove && divMove.step < cliff.step) {
    bullets.push(
      `divergence moved first: ${divMove.div!.toFixed(0)} nats at step ${divMove.step}, ${cliff.step - divMove.step} steps before the first significant pass@1 drop (step ${cliff.step})`,
    );
  }

  const gaps = evals.filter((r) => r.effortGap != null);
  if (gaps.length >= 2) {
    const g0 = gaps[0].effortGap!;
    const dead = gaps.find((g) => g.effortGap! < Math.max(20, g0 * 0.1));
    if (dead)
      bullets.push(
        `effort conditioning collapsed at step ${dead.step} (${g0.toFixed(0)} → ${dead.effortGap!.toFixed(0)} nats)`,
      );
  }

  // first -> last, not global extremes: a transient spike is not the story
  const gt = evals.filter((r) => r.genTok != null && r.genTok > 0);
  if (gt.length >= 2) {
    const a = gt[0].genTok!;
    const b = gt[gt.length - 1].genTok!;
    const ratio = Math.max(a, b) / Math.max(1, Math.min(a, b));
    if (ratio > 3)
      bullets.push(
        `responses went from ${a.toFixed(0)} to ${b.toFixed(0)} tokens (${ratio.toFixed(0)}× ${b < a ? "shorter" : "longer"}) - ${
          b < a
            ? "the model stopped reasoning at length, it did not just get answers wrong"
            : "the model is writing far more per question than it used to"
        }`,
      );
  }

  const tr = evals.filter((r) => r.trunc != null).reduce((a, b) => (b.trunc! > (a?.trunc ?? -1) ? b : a), evals[0]);
  if (tr?.trunc != null && tr.trunc > 0.25)
    bullets.push(
      `truncation peaked at ${(tr.trunc * 100).toFixed(0)}% (step ${tr.step}) - the model was rambling into the token cap`,
    );

  return (
    <div className="card read">
      <div className="mini-label">the read</div>
      <p className="verdict">{verdict}</p>
      {bullets.slice(0, 4).map((b) => (
        <p key={b}>{b}</p>
      ))}
    </div>
  );
}

function MetricStrip({
  d,
  focus,
  onFocus,
  go,
}: {
  d: Derived;
  focus: string | null;
  onFocus: (label: string | null) => void;
  go: (step: number) => void;
}) {
  const cards: [string, string, (r: StepRow) => number | undefined, (v: number) => string][] = [
    ["pass@k", "fraction of questions where at least one of the k samples was correct - the headroom above pass@1", (r) => r.passK, (v) => `${(v * 100).toFixed(1)}%`],
    ["divergence (nats)", "How far the checkpoint's token probabilities moved from base on the baseline's own reasoning traces. ~0 = unchanged policy; very negative = the internals shifted hard - often before accuracy shows it.", (r) => r.div, (v) => v.toFixed(0)],
    ["mean gen tokens", "Average response length. A sudden drop or spike means the model changed how it answers (rambling into the cap, or collapsing to short format-locked answers).", (r) => r.genTok, (v) => v.toFixed(0)],
    ["truncation", "Fraction of samples that hit the max-token cap before finishing. High truncation = the model rambles and never emits a final answer.", (r) => r.trunc, (v) => `${(v * 100).toFixed(0)}%`],
    ["effort gap (nats)", "Log-prob difference between effort=0.9 and effort=0.2 prompts on the same trace. Large = effort conditioning still modulates the model; ~0 = the dial is dead.", (r) => r.effortGap, (v) => v.toFixed(0)],
    ["skip prob", "The model's own probability of ending the response immediately (end-of-message as the first token). Rising = it increasingly wants to emit nothing.", (r) => r.pSkip, (v) => v.toPrecision(2)],
    ["eval cost", "Sampling cost of this eval point, in USD.", (r) => r.cost, (v) => `$${v.toFixed(2)}`],
  ];
  const sel = cards.find(([label]) => label === focus);
  return (
    <>
      <div className="strip">
        {cards.map(([label, tip, get, fmt]) => (
          <div
            className={focus === label ? "mini sel" : "mini"}
            key={label}
            onClick={() => onFocus(focus === label ? null : label)}
          >
            <div className="mini-label"><Tip text={tip}>{label}</Tip></div>
            <TinySeries
              pts={d.rows.filter((r) => get(r) != null).map((r) => ({ x: r.step, y: get(r)! }))}
            />
            <div className="mini-val">
              {(() => {
                const vs = d.rows.filter((r) => get(r) != null);
                return vs.length ? fmt(get(vs[vs.length - 1])!) : "-";
              })()}
            </div>
          </div>
        ))}
      </div>
      {sel && (
        <FocusChart
          key={sel[0]}
          label={sel[0]}
          pts={d.rows.filter((r) => sel[2](r) != null).map((r) => ({ x: r.step, y: sel[2](r)! }))}
          xDomain={[Math.min(...d.rows.map((r) => r.step)), Math.max(...d.rows.map((r) => r.step))]}
          epochStarts={d.epochStarts}
          fmt={sel[3]}
          go={go}
        />
      )}
    </>
  );
}

/** Full-width version of a strip card: labeled axes + clickable eval points. */
function FocusChart({
  label,
  pts,
  xDomain,
  epochStarts,
  fmt,
  go,
}: {
  label: string;
  pts: { x: number; y: number }[];
  xDomain: [number, number];
  epochStarts: number[];
  fmt: (v: number) => string;
  go: (step: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const w = 860;
  const h = 180;
  const pad = 44;
  // share the hero's x range so a metric that stops early (divergence not
  // computed at the last step) shows as a gap, not a stretched line
  const [x0, x1] = xDomain;
  const [y0, y1] = [Math.min(...pts.map((p) => p.y)), Math.max(...pts.map((p) => p.y))];
  const span = Math.max(1e-9, y1 - y0);
  const sx = (x: number) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (w - 2 * pad);
  const sy = (y: number) => h - pad - ((y - y0) / span) * (h - 2 * pad);
  const xticks = tickSteps(x0, x1);
  const epochAt = (s: number) => 1 + epochStarts.filter((e) => e <= s).length;
  const path = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ");
  const hov = hover != null ? pts.find((p) => p.x === hover) : undefined;
  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * w;
    let best = pts[0].x;
    let bd = Infinity;
    for (const p of pts) {
      const dd = Math.abs(sx(p.x) - mx);
      if (dd < bd) {
        bd = dd;
        best = p.x;
      }
    }
    setHover(best);
  };
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      width="100%"
      height="auto"
      preserveAspectRatio="xMidYMid meet"
      className="chart focus"
      onMouseMove={onMove}
      onMouseLeave={() => setHover(null)}
    >
      <line x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
      <line x1={pad} x2={pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
      {xticks.map((x) => (
        <g key={x}>
          <line x1={sx(x)} x2={sx(x)} y1={h - pad} y2={h - pad + 4} style={{ stroke: "var(--muted)" }} strokeOpacity={0.6} />
          <text x={sx(x)} y={h - pad + 16} textAnchor="middle" fontSize={10} style={{ fill: "var(--muted)" }}>{x}</text>
        </g>
      ))}
      {epochStarts.filter((e) => e > x0 && e < x1).map((e) => (
        <line key={e} x1={sx(e)} x2={sx(e)} y1={pad} y2={h - pad} strokeDasharray="2 4" style={{ stroke: "var(--border)" }} />
      ))}
      <path d={path} fill="none" style={{ stroke: "var(--accent)" }} strokeWidth={1.8} />
      {pts.map((p) => (
        <circle
          key={p.x}
          cx={sx(p.x)}
          cy={sy(p.y)}
          r={4.5}
          style={{ fill: "var(--accent)", stroke: "var(--bg)", cursor: "pointer" }}
          strokeWidth={1.5}
          onClick={() => go(p.x)}
        >
          <title>{`step ${p.x} (epoch ${epochAt(p.x)}): ${fmt(p.y)} - click to compare`}</title>
        </circle>
      ))}
      {hov && (
        <g pointerEvents="none">
          <line
            x1={sx(hov.x)}
            x2={sx(hov.x)}
            y1={pad}
            y2={h - pad}
            style={{ stroke: "var(--border)" }}
            strokeDasharray="3 3"
          />
          <circle
            cx={sx(hov.x)}
            cy={sy(hov.y)}
            r={6}
            fill="none"
            style={{ stroke: "var(--accent)" }}
            strokeWidth={1.5}
          />
          <ChartTip
            x={sx(hov.x) > w - 190 ? sx(hov.x) - 176 : sx(hov.x) + 14}
            y={pad + 6}
            lines={[
              [`step ${hov.x} · epoch ${epochAt(hov.x)}`, "var(--text)", true],
              [`${label}  ${fmt(hov.y)}`, "var(--accent)", false],
            ]}
          />
        </g>
      )}
      <text x={6} y={pad} style={{ fill: "var(--muted)" }} fontSize={11}>{fmt(y1)}</text>
      <text x={6} y={h - pad} style={{ fill: "var(--muted)" }} fontSize={11}>{fmt(y0)}</text>
      <text x={w / 2} y={h - 6} textAnchor="middle" style={{ fill: "var(--muted)" }} fontSize={11}>
        optimizer step →
      </text>
      <text x={w - pad} y={16} textAnchor="end" style={{ fill: "var(--muted)" }} fontSize={11}>
        {label}
      </text>
    </svg>
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
        <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={2} style={{ fill: "var(--muted)" }} />
      ))}
      {/* invisible fat dots make the hover values reachable between ticks */}
      {pts.map((p) => (
        <circle key={`h${p.x}`} cx={sx(p.x)} cy={sy(p.y)} r={8} fill="transparent">
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
            <th><Tip text="Accuracy on the frozen eval manifest at this checkpoint.">pass@1</Tip></th>
            <th><Tip text="fraction of questions where at least one of the k samples was correct - the headroom above pass@1">p@k</Tip></th>
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
                <td>{r.pass1 != null ? `${(r.pass1 * 100).toFixed(1)}%` : "-"}</td>
                <td>{r.passK != null ? `${(r.passK * 100).toFixed(1)}%` : "-"}</td>
                <td className={r.delta != null && r.delta < 0 ? "neg" : "pos"}>
                  {r.delta != null
                    ? `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(2)} [${r.ci?.[0].toFixed(2)}, ${r.ci?.[1].toFixed(2)}]`
                    : "-"}
                </td>
                <td>
                  {r.regressions != null ? `${r.regressions}R/${r.gains}G` : "-"}
                </td>
                <td>{r.trunc != null ? `${(r.trunc * 100).toFixed(0)}%` : "-"}</td>
                <td>{r.genTok != null ? r.genTok.toFixed(0) : "-"}</td>
                <td className={r.div != null && r.div < -50 ? "neg" : ""}>
                  {r.div != null ? r.div.toFixed(0) : "-"}
                </td>
                <td>{r.effortGap != null ? r.effortGap.toFixed(0) : "-"}</td>
                <td>{r.cost != null ? `$${r.cost.toFixed(2)}` : "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="sub" style={{ marginTop: 6 }}>- = not computed for this step (divergence and effort run on demand).</p>
    </>
  );
}
