import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { api, type Run, type StudyDetail } from "../api";
import Tip from "../Tip";
import Rollouts from "./Rollouts";

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
    const load = () =>
      api
        .study(study)
        .then((d) => {
          setDetail(d);
          setTouched(new Date());
        })
        .catch((e) => setErr(String(e)));
    load();
    // poll while open so checkpoints eval'd mid-training land without a reload
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [study]);

  const derived = useMemo(() => (detail ? derive(detail) : null), [detail]);
  const [focus, setFocus] = useState<string | null>(null);
  const [touched, setTouched] = useState<Date | null>(null);

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
            {detail.runs.some((r) => r.state === "running") ? " · sampling…" : ""}
            {touched ? ` · live · updated ${touched.toLocaleTimeString()}` : ""}
          </span>
        )}
      </div>
      {err && <p className="err">{err}</p>}
      {detail && derived && derived.rows.length === 0 && (
        <div className="card read">
          <div className="mini-label">{detail.has_rollouts ? "no checkpoint eval series" : "no training metrics"}</div>
          {detail.has_rollouts ? <p>Imported rollouts are shown below. Add checkpoint evals to compare held-out performance.</p> : <p>
            this study has {detail.runs.length} eval run{detail.runs.length === 1 ? "" : "s"} but no
            tracked training loop - there's no step axis to chart. pair runs up in{" "}
            <a href="#/compare">compare</a> or browse them under <a href="#/evals">evals</a>.
          </p>}
        </div>
      )}
      {detail && derived && derived.rows.length > 0 && (
        <>
          <Hero d={derived} go={go} />
          <Narrative d={derived} runs={detail.runs} />
          <MetricStrip d={derived} focus={focus} onFocus={setFocus} go={go} />
        </>
      )}
      {detail?.study === study && detail.has_rollouts && <Rollouts key={study} study={study} />}
      {detail && derived && derived.rows.length > 0 && <StepTable d={derived} runs={detail.runs} />}
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
  trainKey: { key: string; label: string; kind: "loss" | "reward" } | null;
  trainSeries: { step: number; value: number }[];
  rlSeries: {
    key: string;
    label: string;
    tip: string;
    fmt: (v: number) => string;
    pts: { x: number; y: number }[];
  }[];
  epochStarts: number[];
  errorSteps: number[];
}

function derive(detail: StudyDetail): Derived {
  const m = detail.metrics;
  const pick = (suffix: string) => {
    // eval series split across `eval/` (synthesized) and `flipbook/*`
    // (in-loop) namespaces - merge them, flipbook winning a shared step
    const out = new Map<number, number>();
    const keys = Object.keys(m)
      .filter((x) => x.endsWith(suffix) && !x.endsWith("/error"))
      .sort((a, b) => Number(a.startsWith("flipbook/")) - Number(b.startsWith("flipbook/")));
    for (const k of keys) for (const r of m[k]) out.set(r.step, r.value);
    return out;
  };
  const pass1 = pick("/pass1");
  // the evaluator emits pass{n} for n in 2..k; keep the largest n, merged
  const passNs = Object.keys(m)
    .map((x) => Number(/\/pass(\d+)$/.exec(x)?.[1]))
    .filter((n) => n > 1);
  const passK = pick(`/pass${Math.max(0, ...passNs)}`);
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
  // RL logs rollout reward instead - same right-axis role
  const lossKey = ["train_mean_nll", "train_mean_bpb"].find((k) => m[k]?.length) ?? null;
  const rewardKey =
    (m["env/all/reward/total"]?.length ? "env/all/reward/total" : null) ??
    Object.keys(m).find((k) => k.endsWith("/reward/total") && m[k].length) ??
    (m["reward/mean"]?.length ? "reward/mean" : null);
  const trainKey = lossKey
    ? { key: lossKey, label: "train loss", kind: "loss" as const }
    : rewardKey
      ? { key: rewardKey, label: "rollout reward", kind: "reward" as const }
      : null;
  const trainSeries = trainKey ? m[trainKey.key] : [];

  const rlSeries = (
    [
      ["env/all/reward/total", "reward", "mean verifiable reward per rollout batch - what the optimizer is actually maximizing", (v: number) => v.toFixed(2)],
      ["env/all/by_group/frac_mixed", "mixed groups", "fraction of rollout groups whose rewards differed within the group - the share carrying any gradient signal. Only meaningful for binary rewards; a shaped reward makes nearly every group 'mixed'", (v: number) => `${(v * 100).toFixed(0)}%`],
      ["env/all/ac_tokens_per_turn", "rollout tokens", "action tokens generated per turn during rollouts - drift means the policy changed how much it says on the training distribution", (v: number) => v.toFixed(0)],
      ["optim/entropy", "entropy", "policy entropy over sampled actions - collapse means the policy sharpens onto fewer tokens", (v: number) => v.toFixed(3)],
      ["kl_pre_post_v1", "kl pre/post", "KL of the policy before vs after each optim step - the step size the optimizer actually took", (v: number) => (Math.abs(v) < 1e-3 ? v.toExponential(1) : v.toFixed(4))],
    ] as const
  )
    .filter(([key]) => m[key]?.length)
    .map(([key, label, tip, fmt]) => ({
      key,
      label,
      tip,
      fmt,
      pts: m[key].map((r) => ({ x: r.step, y: r.value })),
    }));

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

  // in-loop eval failures are logged as a metric, not an exception - surface them
  const errorSteps = [
    ...new Set(
      Object.keys(m)
        .filter((x) => x.endsWith("/error"))
        .flatMap((x) => m[x].filter((r) => r.value > 0).map((r) => r.step)),
    ),
  ].sort((a, b) => a - b);

  const steps = [...new Set([...pass1.keys(), ...trainSeries.map((s) => s.step)])].sort(
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
    loss: trainSeries.find((s) => s.step === step)?.value,
    effortGap: gapByStep.get(step),
  }));
  return { rows, metricKeys: Object.keys(m), trainKey, trainSeries, rlSeries, epochStarts, errorSteps };
}

/** pass@1 vs the train signal: loss on its own right axis for SFT, rollout
 *  reward sharing the 0-1 eval axis for RL - the collision chart. */
function Hero({ d, go }: { d: Derived; go: (step: number) => void }) {
  const [hover, setHover] = useState<number | null>(null);
  const pts = d.rows.filter((r) => r.pass1 != null);
  // rewards are correctness fractions: they share the [0,1] eval axis
  const isReward = d.trainKey?.kind === "reward";
  const lossMax = Math.max(1e-9, ...d.trainSeries.map((s) => s.value));
  if (!pts.length) return <p className="sub">no eval metrics for this study</p>;

  const w = 860;
  const h = 280;
  const pad = 44;
  const allSteps = [
    ...pts.map((r) => r.step),
    ...d.trainSeries.map((s) => s.step),
  ];
  const x0 = Math.min(...allSteps);
  const x1 = Math.max(...allSteps);
  const sx = (x: number) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (w - 2 * pad);
  const xticks = tickSteps(x0, x1);
  const syPass = (y: number) => h - pad - Math.max(0, Math.min(1, y)) * (h - 2 * pad);
  // 1-indexed data pass containing this step (boundaries are where the
  // logged epoch counter ticks up)
  const epochAt = (s: number) => 1 + d.epochStarts.filter((e) => e <= s).length;
  const syLoss = isReward
    ? syPass
    : (y: number) => h - pad - (y / lossMax) * (h - 2 * pad);
  // signed percentage points, for CI text
  const pp = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}`;

  const passPath = pts.map((r, i) => `${i ? "L" : "M"}${sx(r.step)},${syPass(r.pass1!)}`).join(" ");
  const lossPath = d.trainSeries
    .map((r, i) => `${i ? "L" : "M"}${sx(r.step)},${syLoss(r.value)}`)
    .join(" ");

  // snap to every logged step - evals are sparse, loss lands every step
  const hoverSteps = [
    ...new Set([...pts.map((r) => r.step), ...d.trainSeries.map((s) => s.step)]),
  ].sort((a, b) => a - b);
  const hovRow = hover != null ? pts.find((r) => r.step === hover) : undefined;
  const hovLoss = hover != null ? d.trainSeries.find((s) => s.step === hover)?.value : undefined;
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
        preserveAspectRatio="xMidYMid meet"
        className="chart"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <line x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        <line x1={pad} x2={pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        {!isReward && (
          <line x1={w - pad} x2={w - pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        )}
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
        {/* in-loop eval errors land on the x axis as hollow markers */}
        {d.errorSteps.map((s) => (
          <circle
            key={`err${s}`}
            cx={sx(s)}
            cy={h - pad - 8}
            r={4.5}
            fill="none"
            style={{ stroke: "var(--neg)" }}
            strokeWidth={1.4}
          >
            <title>{`in-loop eval errored at step ${s}`}</title>
          </circle>
        ))}
        {lossPath && (
          <path d={lossPath} fill="none" style={{ stroke: "var(--accent2)" }} strokeWidth={1.4} strokeDasharray="4 3" />
        )}
        {/* paired-CI whiskers on eval points, under the line so dots stay readable */}
        {pts
          .filter((r) => r.delta != null && r.ci)
          .map((r) => {
            const lo = Math.max(0, Math.min(1, r.pass1! - (r.delta! - r.ci![0])));
            const hi = Math.max(0, Math.min(1, r.pass1! + (r.ci![1] - r.delta!)));
            const x = sx(r.step);
            return (
              <g
                key={`ci${r.step}`}
                style={{ stroke: "var(--accent)" }}
                strokeOpacity={0.45}
                strokeWidth={1.2}
              >
                <line x1={x} x2={x} y1={syPass(lo)} y2={syPass(hi)} />
                <line x1={x - 3} x2={x + 3} y1={syPass(lo)} y2={syPass(lo)} />
                <line x1={x - 3} x2={x + 3} y1={syPass(hi)} y2={syPass(hi)} />
                <title>{`step ${r.step}: 95% paired CI on delta vs base [${pp(r.ci![0])}, ${pp(r.ci![1])}] pp`}</title>
              </g>
            );
          })}
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
                  ? ([[`pass@1 ${(hovRow.pass1! * 100).toFixed(1)}%${hovRow.delta != null && hovRow.ci ? `  CI [${pp(hovRow.ci[0])}, ${pp(hovRow.ci[1])}] pp` : ""}`, "var(--accent)", false]] as [string, string, boolean][])
                  : []),
                ...(hovLoss != null
                  ? ([[`${isReward ? "reward" : "nll"}  ${isReward ? hovLoss.toFixed(2) : hovLoss.toPrecision(3)}`, "var(--accent2)", false]] as [string, string, boolean][])
                  : []),
              ]}
            />
          </g>
        )}
        <text x={6} y={pad} style={{ fill: "var(--accent)" }} fontSize={11}>100%</text>
        <text x={6} y={h - pad} style={{ fill: "var(--accent)" }} fontSize={11}>0%</text>
        {d.trainSeries.length > 0 && !isReward && (
          <>
            <text x={w - pad + 6} y={pad} style={{ fill: "var(--accent2)" }} fontSize={11}>{lossMax.toPrecision(2)}</text>
            <text x={w - pad + 6} y={h - pad} style={{ fill: "var(--accent2)" }} fontSize={11}>0</text>
          </>
        )}
        <text x={w / 2} y={h - 4} textAnchor="middle" style={{ fill: "var(--muted)" }} fontSize={11}>
          optimizer step →
        </text>
      </svg>
      <div className="legend">
        <Tip text="Fraction of frozen eval questions answered correctly at this checkpoint (left axis). Click a point to compare that step against baseline.">
          <span style={{ color: "var(--accent)" }}>━ pass@1 (click a point)</span>
        </Tip>
        {pts.some((r) => r.delta != null && r.ci) && (
          <Tip text="Bootstrap 95% CI of the paired per-question delta vs baseline, drawn around each point. If the whisker crosses the baseline's level, the change is within noise.">
            <span style={{ color: "var(--accent)" }}>┃ 95% CI</span>
          </Tip>
        )}
        {d.trainKey && (
          <Tip text={
            isReward
              ? `${d.trainKey.key}: mean rollout reward on the training env, same 0-1 axis as pass@1 - what the optimizer sees vs what you care about. It can saturate while held-out eval never moves`
              : `${d.trainKey.key}: training loss on the fine-tuning batches - what the optimizer sees (right axis). It can keep falling while eval behavior collapses; that's the collision this chart exists to show. Logged every optimizer step (one batch); evals run every ${evalGap ?? "?"} steps.`
          }>
            <span style={{ color: "var(--accent2)" }}>┅ {d.trainKey.label}</span>
          </Tip>
        )}
      </div>
      <p className="sub" style={{ marginTop: 4 }}>
        x = optimizer step (one training batch).
        {d.trainSeries.length ? ` ${d.trainKey?.label} is logged every step; ` : " "}
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
function Narrative({ d, runs }: { d: Derived; runs: Run[] }) {
  const evals = d.rows.filter((r) => r.pass1 != null);
  if (evals.length < 2) return null;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const bullets: { t: string; href?: string }[] = [];
  // every claim should be one click from its evidence
  const base = runs.find((r) => r.train_step == null);
  const atStep = (s: number) => runs.find((r) => r.train_step === s)?.run_id;
  const cmp = (s: number) =>
    base && atStep(s) ? `#/compare?a=${base.run_id}&b=${atStep(s)}` : undefined;
  const div = (s: number) =>
    base && atStep(s) ? `#/divergence?base=${base.run_id}&ckpt=${atStep(s)}` : undefined;

  const peak = evals.reduce((a, b) => (b.pass1! > a.pass1! ? b : a));
  const after = evals.filter((r) => r.step >= peak.step);
  const trough = after.reduce((a, b) => (b.pass1! < a.pass1! ? b : a));
  const last = evals[evals.length - 1];
  const first = evals[0];
  // a moved point estimate only counts when its paired CI excludes zero;
  // steps without a CI fall back to the point-estimate rule
  const dropCi = trough.ci;
  const endCi = last.ci;
  const collapsed =
    peak.pass1! - trough.pass1! > 0.05 &&
    last.pass1! < peak.pass1! - 0.05 &&
    (dropCi ? dropCi[1] < 0 : true);
  const improved =
    last.pass1! - first.pass1! > 0.05 && (endCi ? endCi[0] > 0 : true);
  const noisy =
    !collapsed && !improved && endCi != null && endCi[0] <= 0 && endCi[1] >= 0 &&
    Math.abs(last.pass1! - first.pass1!) >= 0.05;
  // RL studies: reward can move independently of held-out eval - that's the read
  const rewardPts = d.rlSeries.find((s) => s.key.endsWith("/reward/total"))?.pts;
  const tokPts = d.rlSeries.find((s) => s.key === "env/all/ac_tokens_per_turn")?.pts;
  const fracPts = d.rlSeries.find((s) => s.key === "env/all/by_group/frac_mixed")?.pts;
  const fracMean = fracPts?.length
    ? fracPts.reduce((a, p) => a + p.y, 0) / fracPts.length
    : null;
  const rlBullets: { t: string; href?: string }[] = [];
  let rewardRose = false;
  let noSignal = false;
  if (rewardPts?.length) {
    const rf = rewardPts[0].y;
    const rl = rewardPts[rewardPts.length - 1].y;
    rewardRose = rl - rf > 0.1;
    if (fracPts && fracMean != null && fracMean < 0.1) {
      // saturated or uniform groups carry zero advantage either way - the
      // optimizer had nothing to learn from, which is worse news than a drop
      noSignal = true;
      rlBullets.push({
        t: `only ${(fracMean * 100).toFixed(0)}% of rollout groups were mixed (reward ${rf.toFixed(2)} -> ${rl.toFixed(2)}): group-centred advantages were ~0, the optimizer had almost no signal. Change the env difficulty before spending more`,
      });
    } else if (rf >= 0.95)
      rlBullets.push({
        t: `reward saturated from the first batch (${rf.toFixed(2)}): this env gives the policy no gradient signal`,
      });
    else if (rewardRose && collapsed)
      rlBullets.push({
        t: `rollout reward rose ${rf.toFixed(2)} -> ${rl.toFixed(2)} while held-out pass@1 fell: the policy is fitting the reward, not the capability`,
      });
    else if (rewardRose && improved)
      rlBullets.push({
        t: `reward and held-out pass@1 rose together (${rf.toFixed(2)} -> ${rl.toFixed(2)}, ${pct(first.pass1!)} -> ${pct(last.pass1!)})`,
      });
    else if (rewardRose)
      rlBullets.push({
        t: `rollout reward rose ${rf.toFixed(2)} -> ${rl.toFixed(2)}; held-out pass@1 unchanged: no transfer to the eval either way`,
      });
    else rlBullets.push({ t: `rollout reward flat (${rf.toFixed(2)} -> ${rl.toFixed(2)})` });
  }
  if (d.errorSteps.length)
    rlBullets.push({
      t: `in-loop eval errored at step${d.errorSteps.length > 1 ? "s" : ""} ${d.errorSteps.join(", ")} - training continued, check the run log (metrics.jsonl)`,
    });
  if (tokPts && tokPts.length >= 2) {
    const tf = tokPts[0].y;
    const tl = tokPts[tokPts.length - 1].y;
    if (Math.abs(tl - tf) / tf > 0.2)
      rlBullets.push({
        t: `rollouts ${tl < tf ? "shortened" : "lengthened"} ${tf.toFixed(0)} -> ${tl.toFixed(0)} tokens per turn on the training distribution`,
      });
  }
  const hacking = rewardRose && collapsed;
  const verdict = hacking
    ? "reward up, held-out eval down"
    : collapsed
      ? "the fine-tune regressed this eval"
      : improved
        ? "the fine-tune improved this eval"
        : noSignal
          ? "no learning signal, no eval-visible change"
          : noisy || peak.pass1! - trough.pass1! <= 0.05
            ? "no eval-visible change"
            : "pass@1 dipped mid-training and recovered";
  // did divergence move before the first significant accuracy drop?
  const cliff = evals.find((r) => r.ci && r.ci[1] < 0);
  if (noisy) {
    const sgn = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
    bullets.push({
      t: `pass@1 ${pct(first.pass1!)} -> ${pct(last.pass1!)} by step ${last.step}, but the CI [${sgn(endCi![0])}, ${sgn(endCi![1])}] includes zero - within noise`,
      href: cmp(last.step),
    });
  } else if (peak.pass1! - trough.pass1! > 0.05) {
    // a "peak" inside the CI of the first point is noise - anchor on where it held
    const realPeak = peak.step !== first.step && peak.ci && peak.ci[0] > 0;
    const held = cliff ? evals[Math.max(0, evals.indexOf(cliff) - 1)] : peak;
    const open = realPeak
      ? `pass@1 rose to ${pct(peak.pass1!)} (step ${peak.step})`
      : held.step === first.step
        ? `pass@1 started at ${pct(first.pass1!)} (step ${first.step})`
        : `pass@1 held near ${pct(first.pass1!)} through step ${held.step}`;
    bullets.push({
      t:
        `${open}, then fell to ${pct(trough.pass1!)} by step ${trough.step}` +
        (last.pass1! > trough.pass1! + 0.02 ? ` - ending at ${pct(last.pass1!)} (step ${last.step})` : ` and stayed there`),
      href: cmp(trough.step),
    });
  } else {
    bullets.push({
      t: `pass@1 stayed flat through training (${pct(evals[0].pass1!)} → ${pct(last.pass1!)}) - no eval-visible regression`,
      href: cmp(last.step),
    });
  }

  const divMove = evals.find((r) => r.div != null && r.div < -50);
  if (cliff && divMove && divMove.step < cliff.step) {
    bullets.push({
      t: `divergence moved first: ${divMove.div!.toFixed(0)} nats at step ${divMove.step}, ${cliff.step - divMove.step} steps before the first significant pass@1 drop (step ${cliff.step})`,
      href: div(divMove.step),
    });
  }

  const gaps = evals.filter((r) => r.effortGap != null);
  if (gaps.length >= 2) {
    const g0 = gaps[0].effortGap!;
    const dead = gaps.find((g) => g.effortGap! < Math.max(20, g0 * 0.1));
    if (dead)
      bullets.push({
        t: `effort conditioning collapsed at step ${dead.step} (${g0.toFixed(0)} → ${dead.effortGap!.toFixed(0)} nats)`,
        href: cmp(dead.step),
      });
  }

  // first -> last, not global extremes: a transient spike is not the story
  const gt = evals.filter((r) => r.genTok != null && r.genTok > 0);
  if (gt.length >= 2) {
    const a = gt[0].genTok!;
    const b = gt[gt.length - 1].genTok!;
    const ratio = Math.max(a, b) / Math.max(1, Math.min(a, b));
    if (ratio > 3)
      bullets.push({
        t: `responses went from ${a.toFixed(0)} to ${b.toFixed(0)} tokens (${ratio.toFixed(0)}× ${b < a ? "shorter" : "longer"}) - ${
          b < a
            ? "the model stopped reasoning at length, it did not just get answers wrong"
            : "the model is writing far more per question than it used to"
        }`,
        href: cmp(gt[gt.length - 1].step),
      });
    else if (b < a * 0.7) {
      // a gradual thinning of the reasoning is a leading indicator too:
      // name the step it crossed -20% and whether that preceded the cliff
      const onset = gt.find((r) => r.genTok! < a * 0.8);
      const lead = onset && cliff && onset.step < cliff.step ? `, ${cliff.step - onset.step} steps before the first significant pass@1 drop` : "";
      bullets.push({
        t: `held-out responses thinned ${a.toFixed(0)} -> ${b.toFixed(0)} tokens (${((1 - b / a) * 100).toFixed(0)}% shorter), first past -20% at step ${onset!.step}${lead} - the model reasons less on problems it never trained on`,
        href: cmp(onset!.step),
      });
    }
  }

  const tr = evals.filter((r) => r.trunc != null).reduce((a, b) => (b.trunc! > (a?.trunc ?? -1) ? b : a), evals[0]);
  if (tr?.trunc != null && tr.trunc > 0.25)
    bullets.push({
      t: `truncation peaked at ${(tr.trunc * 100).toFixed(0)}% (step ${tr.step}) - the model was rambling into the token cap`,
      href: cmp(tr.step),
    });

  return (
    <div className="card read">
      <div className="mini-label">the read</div>
      <p className="verdict">{verdict}</p>
      {[...rlBullets, ...bullets].slice(0, 5).map((b) =>
        b.href ? (
          <p key={b.t}>
            <a href={b.href}>{b.t}</a>
          </p>
        ) : (
          <p key={b.t}>{b.t}</p>
        ),
      )}
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
  const evalCards: [string, string, (r: StepRow) => number | undefined, (v: number) => string][] = [
    ["pass@k", "fraction of questions where at least one of the k samples was correct - the headroom above pass@1", (r) => r.passK, (v) => `${(v * 100).toFixed(1)}%`],
    ["divergence (nats)", "How far the checkpoint's token probabilities moved from base on the baseline's own reasoning traces. Positive = the checkpoint assigns the baseline's traces higher probability than the baseline did (sharpened onto its own distribution); negative = it moved away from them. ~0 = unchanged policy; very negative = the internals shifted hard - often before accuracy shows it.", (r) => r.div, (v) => v.toLocaleString("en-US", { maximumFractionDigits: 0 })],
    ["mean gen tokens", "Average response length. A sudden drop or spike means the model changed how it answers (rambling into the cap, or collapsing to short format-locked answers).", (r) => r.genTok, (v) => v.toLocaleString("en-US", { maximumFractionDigits: 0 })],
    ["truncation", "Fraction of samples that hit the max-token cap before finishing. High truncation = the model rambles and never emits a final answer.", (r) => r.trunc, (v) => `${(v * 100).toFixed(0)}%`],
    ["effort gap (nats)", "Log-prob difference between effort=0.9 and effort=0.2 prompts on the same trace. Large = effort conditioning still modulates the model; ~0 = the dial is dead.", (r) => r.effortGap, (v) => v.toLocaleString("en-US", { maximumFractionDigits: 0 })],
    ["skip prob", "The model's own probability of ending the response immediately (end-of-message as the first token). Rising = it increasingly wants to emit nothing.", (r) => r.pSkip, (v) => `${(v * 100).toFixed(1)}%`],
    ["eval cost", "Sampling cost of this eval point, in USD.", (r) => r.cost, (v) => `$${v.toFixed(2)}`],
  ];
  const evalGroup = evalCards
    .map(([label, tip, get, fmt]) => ({
      label,
      tip,
      fmt,
      pts: d.rows.filter((r) => get(r) != null).map((r) => ({ x: r.step, y: get(r)! })),
    }))
    .filter((c) => c.pts.length >= 2);
  const rlGroup = d.rlSeries.filter((c) => c.pts.length >= 2);
  const sel = [...evalGroup, ...rlGroup].find((c) => c.label === focus);
  const card = (c: (typeof evalGroup)[number]) => (
    <div
      className={focus === c.label ? "mini sel" : "mini"}
      key={c.label}
      onClick={() => onFocus(focus === c.label ? null : c.label)}
    >
      <div className="mini-label"><Tip text={c.tip}>{c.label}</Tip></div>
      <TinySeries pts={c.pts} />
      <div className="mini-val">{c.fmt(c.pts[c.pts.length - 1].y)}</div>
    </div>
  );
  return (
    <>
      <div className="strip-group">
        <div className="mini-label">held-out eval</div>
        <div className="strip">{evalGroup.map(card)}</div>
      </div>
      {rlGroup.length > 0 && (
        <div className="strip-group">
          <div className="mini-label">training</div>
          <div className="strip">{rlGroup.map(card)}</div>
        </div>
      )}
      {sel && (
        <FocusChart
          key={sel.label}
          label={sel.label}
          pts={sel.pts}
          ghost={d.rows.filter((r) => r.pass1 != null).map((r) => ({ x: r.step, y: r.pass1! }))}
          ghostLabel="pass@1"
          xDomain={[Math.min(...d.rows.map((r) => r.step)), Math.max(...d.rows.map((r) => r.step))]}
          epochStarts={d.epochStarts}
          fmt={sel.fmt}
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
  ghost,
  ghostLabel,
  xDomain,
  epochStarts,
  fmt,
  go,
}: {
  label: string;
  pts: { x: number; y: number }[];
  /** pass@1 drawn faintly behind the focused metric - the reference every
   *  other metric is read against; kept on its own [0,1] scale */
  ghost?: { x: number; y: number }[];
  ghostLabel?: string;
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
  // an empty series has no y domain - render the frame, not Infinity axes
  if (pts.length === 0) {
    return (
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" className="chart focus">
        <line x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        <line x1={pad} x2={pad} y1={pad} y2={h - pad} style={{ stroke: "var(--border)" }} />
        <text x={w / 2} y={h / 2} textAnchor="middle" style={{ fill: "var(--muted)" }} fontSize={11}>
          no {label} computed yet
        </text>
        <text x={w - pad} y={16} textAnchor="end" style={{ fill: "var(--muted)" }} fontSize={11}>
          {label}{ghostLabel ? `  ·  ┅ ${ghostLabel}` : ""}
        </text>
      </svg>
    );
  }
  const [y0, y1] = [Math.min(...pts.map((p) => p.y)), Math.max(...pts.map((p) => p.y))];
  const span = Math.max(1e-9, y1 - y0);
  const sx = (x: number) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (w - 2 * pad);
  const sy = (y: number) => h - pad - ((y - y0) / span) * (h - 2 * pad);
  const xticks = tickSteps(x0, x1);
  const epochAt = (s: number) => 1 + epochStarts.filter((e) => e <= s).length;
  const path = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ");
  // ghost (pass@1) is always [0,1] so it shares the plot height directly
  const gsy = (y: number) => h - pad - Math.max(0, Math.min(1, y)) * (h - 2 * pad);
  const ghostPath =
    ghost && ghost.length > 1
      ? ghost.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${gsy(p.y)}`).join(" ")
      : null;
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
      {ghostPath && (
        <path d={ghostPath} fill="none" style={{ stroke: "var(--muted)" }} strokeWidth={1.4} strokeDasharray="4 3" opacity={0.6} />
      )}
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
              ...(ghost && ghostLabel
                ? (() => {
                    const g = ghost.find((p) => p.x === hov.x);
                    return g
                      ? ([[`${ghostLabel}  ${(g.y * 100).toFixed(1)}%`, "var(--muted)", false]] as [string, string, boolean][])
                      : [];
                  })()
                : []),
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
        {label}{ghostLabel ? `  ·  ┅ ${ghostLabel}` : ""}
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
            <th><Tip text="Mean per-token log-prob shift of the checkpoint vs base, scored on the baseline's own traces. Positive = the checkpoint assigns the baseline's traces higher probability than the baseline did (sharpened onto its own distribution); negative = it moved away from them.">div nats</Tip></th>
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
                {/* colour only when the CI excludes zero; a delta inside the noise band is not a verdict */}
                <td className={r.ci && r.ci[1] < 0 ? "neg" : r.ci && r.ci[0] > 0 ? "pos" : undefined}>
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
          {runs
            .filter((r) => r.state === "running" && r.train_step != null)
            .map((r) => (
              <tr key={r.run_id}>
                <td>step {r.train_step}</td>
                <td colSpan={8} className="sub">
                  sampling… this eval is still running; the page refreshes every 8s
                </td>
              </tr>
            ))}
        </tbody>
      </table>
      <p className="sub" style={{ marginTop: 6 }}>- = not computed for this step (divergence and effort run on demand).</p>
    </>
  );
}
