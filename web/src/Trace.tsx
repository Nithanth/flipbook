import { useEffect, useRef, useState } from "react";
import { api, type TokenTrace, type BranchResult } from "./api";

// same cumulative-nats threshold the backend's divergence_pos uses (diverge.TAU)
const DIVERGE_TAU = 5.0;

// renderer sentinels like <|content_thinking|> or {lt}x{gt} are real scored
// tokens, but they're structure, not content - dim them
const CTL = /^(<\|[^|]+\|>|\{lt\}.*\{gt\})$/;

/** First token index where the cumulative Δlogprob crosses -TAU, else null. */
export function firstDivPos(delta: number[]): number | null {
  let cum = 0;
  for (let i = 0; i < delta.length; i++) {
    cum += delta[i];
    if (cum <= -DIVERGE_TAU) return i;
  }
  return null;
}

/** Baseline trace text heat-mapped by per-token Δlogprob, with its legend.
 *  onPick/cut wire "click a token to set the branch cut" through the parent. */
export function Trace({
  tokens,
  reason,
  mark,
  onPick,
  cut,
}: {
  tokens: { t: string; d: number }[];
  reason?: string;
  mark?: number | null;
  onPick?: (i: number) => void;
  cut?: number | null;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [windowStart, setWindowStart] = useState<number | null>(null);
  const max = tokens.reduce((m, t) => Math.max(m, Math.abs(t.d)), 1e-9);
  const longTrace = tokens.length > 160;
  const binN = Math.max(8, Math.ceil(tokens.length / 160));
  const windows = [];
  if (longTrace) {
    for (let start = 0; start < tokens.length; start += binN) {
      const segment = tokens.slice(start, start + binN);
      const sum = segment.reduce((total, token) => total + token.d, 0);
      windows.push({ start, sum, mean: sum / segment.length });
    }
  }
  const hot: typeof windows = [];
  for (const w of windows.sort((a, b) => Math.abs(b.sum) - Math.abs(a.sum))) {
    if (Math.abs(w.sum) < 5) break;
    if (hot.every(other => Math.abs(other.start - w.start) >= binN * 2)) hot.push(w);
    if (hot.length === 3) break;
  }
  function jump(start: number) {
    setWindowStart(start);
    const el = wrapRef.current?.querySelector(`[data-tok="${start}"]`);
    if (el instanceof HTMLElement && wrapRef.current) {
      wrapRef.current.scrollTop = el.offsetTop - wrapRef.current.clientHeight / 2;
    }
  }
  return (
    <>
      <div className="sub" style={{ marginBottom: 6 }}>
        {reason && <p>{reason} Numbers below are token positions.</p>}
        Baseline response · <span style={{ color: "var(--neg)" }}>Red: less likely</span>
        {" · "}<span style={{ color: "var(--accent)" }}>Blue: more likely</span> under the checkpoint
        {mark != null && ` · First divergence: token ${mark} (underlined)`}
        {onPick && " · Click a token to set the branch cut"}
      </div>
      <div ref={wrapRef} className="traceread">
        {tokens.map((t, i) => {
          const a = Math.sqrt(Math.abs(t.d) / max);
          return (
            <span
              key={i}
              data-tok={i % binN === 0 ? i : undefined}
              className={
                `${i === mark ? "divmark" : ""}${onPick && i === cut ? " cutmark" : ""}${windowStart != null && i >= windowStart && i < windowStart + binN ? " jumpmark" : ""}${
                  CTL.test(t.t) ? " ctl" : ""
                }${onPick ? " pickable" : ""}` || undefined
              }
              style={{
                background: `color-mix(in srgb, ${
                  t.d < 0 ? "var(--neg)" : "var(--accent)"
                } ${(a * 80).toFixed(0)}%, transparent)`,
              }}
              title={`token ${i}${i === mark ? " · first divergence" : ""}: Δ ${t.d.toFixed(3)} nats${onPick ? " - click to cut here" : ""}`}
              onClick={onPick ? () => onPick(i) : undefined}
            >
              {t.t}
            </span>
          );
        })}
      </div>
      {hot.length > 0 && (
        <div className="sub hotchips">
          Largest shifts
          {hot.map(w => {
            const excerpt = tokens.slice(w.start, w.start + binN).map(t => t.t).join("").trim();
            return <button key={w.start} className="hotchip"
              aria-pressed={windowStart === w.start}
              title={`Tokens ${w.start}–${Math.min(w.start + binN, tokens.length) - 1} · total ${w.sum.toFixed(2)} nats · mean ${w.mean.toFixed(2)} nats/token`}
              onClick={() => jump(w.start)}>
              {w.mean < 0 ? "Less likely" : "More likely"} · {excerpt.length > 48 ? excerpt.slice(0, 48) + "…" : excerpt}
            </button>;
          })}
        </div>
      )}
    </>
  );
}

/** The baseline's actual trace text, heat-mapped by per-token Δlogprob. Uses numbered positions when only the scores are available. */
export function TraceView({
  base,
  ckpt,
  row,
  sample,
  delta,
  mark,
  onPick,
  cut,
}: {
  base: string;
  ckpt: string;
  row: string;
  sample: number;
  delta: number[];
  mark: number | null;
  onPick?: (i: number) => void;
  cut?: number | null;
}) {
  const [trace, setTrace] = useState<TokenTrace | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    setTrace(null);
    setError("");
    api.divergenceTrace(base, ckpt, row, sample)
      .then(result => { if (active) setTrace(result); })
      .catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [base, ckpt, row, sample]);

  if (error) return <><p className="err">Could not load token trace: {error}</p><Spark delta={delta} mark={null} w={800} h={72} /></>;
  if (!trace) return <p className="sub">loading trace…</p>;
  return <Trace tokens={traceTokens(trace)} reason={trace.reason} mark={mark} onPick={onPick} cut={cut} />;
}

export function traceTokens(trace: TokenTrace) {
  return trace.delta?.map((d, i) => ({ t: `${i} `, d })) ?? trace.tokens;
}

/** Delta sparkline: red = ckpt less confident than base, blue = more.
 *  `n` is the real token count when `delta` arrived downsampled — it scales
 *  the first-divergence tick. */
export function Spark({
  delta,
  mark,
  w = 220,
  h = 28,
  n,
  scale,
}: {
  delta: number[];
  mark: number | null;
  w?: number;
  h?: number;
  n?: number;
  /** shared |delta| ceiling (e.g. pair p95) - without it each row
      self-normalizes and bar heights stop comparing across rows */
  scale?: number;
}) {
  if (!delta.length) return null;
  const step = Math.max(1, Math.floor(delta.length / w));
  const pts: number[] = [];
  for (let i = 0; i < delta.length; i += step) pts.push(delta[i]);
  const max = Math.max(1e-9, scale ?? Math.max(1e-9, ...pts.map(Math.abs)));
  const bw = w / pts.length;
  return (
    <svg width={w} height={h} className="spark">
      <line x1={0} x2={w} y1={h / 2} y2={h / 2} style={{ stroke: "var(--border)" }} />
      {pts.map((d, i) => {
        const bh = Math.min(h / 2, (Math.abs(d) / max) * (h / 2));
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
          x1={(mark / (n ?? delta.length)) * w}
          x2={(mark / (n ?? delta.length)) * w}
          y1={0}
          y2={h}
          style={{ stroke: "var(--warn)" }}
          strokeWidth={2}
        />
      )}
    </svg>
  );
}

/** "What would the ckpt do instead?" - greedy ckpt sample from a cut on the
 *  base's trace. The only paid interaction in the UI (~96 tokens/click). */
export function BranchView({
  base,
  ckpt,
  row,
  sample,
  pos,
  nTokens,
}: {
  base: string;
  ckpt: string;
  row: string;
  sample: number;
  pos: number | null;
  nTokens: number;
}) {
  const [at, setAt] = useState(String(pos ?? 0));
  const [res, setRes] = useState<BranchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setAt(String(pos ?? 0));
    setRes(null);
    setErr(null);
  }, [base, ckpt, row, sample, pos]);

  const run = () => {
    const p = Math.max(0, Math.min(parseInt(at || "0", 10) || 0, nTokens - 1));
    setBusy(true);
    setErr(null);
    api
      .divergenceBranch(base, ckpt, row, sample, p)
      .then(setRes)
      .catch((e) => setErr(String(e)))
      .finally(() => setBusy(false));
  };

  return (
    <div className="branch">
      <div className="row">
        <label className="sub">
          cut at token{" "}
          <input
            className="pos-in"
            type="number"
            min={0}
            max={nTokens - 1}
            value={at}
            onChange={(e) => setAt(e.target.value)}
          />
        </label>
        <button className="theme-btn" onClick={run} disabled={busy}>
          {busy ? "sampling…" : "branch ckpt here"}
        </button>
        <span className="sub">greedy ckpt continuation, ~96 tokens - a paid call</span>
        {err && <span className="err">{err}</span>}
      </div>
      {res && (
        <>
          <div className="sub" style={{ marginTop: 8 }}>
            base said <code>{JSON.stringify(res.base_tok)}</code> (lp{" "}
            {res.base_tok_lp.toFixed(2)}) · ckpt scores that token{" "}
            {res.ckpt_lp_on_base_tok.toFixed(2)} · ckpt instead emits{" "}
            <code>{JSON.stringify(res.ckpt_first_tok)}</code>
            {res.ckpt_first_tok_lp != null
              ? ` (lp ${res.ckpt_first_tok_lp.toFixed(2)})`
              : ""}
          </div>
          <div className="branchcols">
            <div>
              <div className="mini-label">base continued</div>
              <pre>{res.base_cont || "(empty)"}</pre>
            </div>
            <div>
              <div className="mini-label">ckpt from the same prefix (greedy)</div>
              <pre>{res.ckpt_cont || "(empty)"}</pre>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
