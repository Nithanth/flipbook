import { useEffect, useRef, useState } from "react";
import { api, type BranchResult } from "./api";

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

/** Baseline trace text heat-mapped by per-token Δlogprob, with its legend. */
export function Trace({
  tokens,
  mark,
}: {
  tokens: { t: string; d: number }[];
  mark?: number | null;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const max = Math.max(1e-9, ...tokens.map((t) => Math.abs(t.d)));
  // long traces wash out to uniform red - a windowed-mean strip above the text
  // keeps regional structure legible; clicking a bin scrolls the text to it
  const binN = Math.max(1, Math.ceil(tokens.length / 160));
  const bins: { start: number; mean: number }[] = [];
  for (let i = 0; i < tokens.length; i += binN) {
    const seg = tokens.slice(i, i + binN);
    bins.push({ start: i, mean: seg.reduce((a, t) => a + t.d, 0) / seg.length });
  }
  const binMax = Math.max(1e-9, ...bins.map((b) => Math.abs(b.mean)));
  const jump = (start: number) => {
    const el = wrapRef.current?.querySelector(`[data-tok="${start}"]`);
    if (el instanceof HTMLElement && wrapRef.current) {
      wrapRef.current.scrollTop = el.offsetTop - wrapRef.current.clientHeight / 2;
    }
  };
  return (
    <>
      <div className="sub" style={{ marginBottom: 6 }}>
        baseline trace colored by Δlogprob per token -{" "}
        <span style={{ color: "var(--neg)" }}>red</span> = ckpt less confident,{" "}
        <span style={{ color: "var(--accent)" }}>blue</span> = more confident
        {mark != null ? " · amber edge = first divergence" : ""}
        {" · strip = mean Δ per window, click to jump"}
      </div>
      <div className="binstrip">
        {bins.map((b, i) => (
          <div
            key={i}
            className={`bin${mark != null && mark >= b.start && mark < b.start + binN ? " binmark" : ""}`}
            title={`tokens ${b.start}-${Math.min(b.start + binN, tokens.length)}: mean Δ ${b.mean.toFixed(3)} nats`}
            onClick={() => jump(b.start)}
            style={{
              background: `color-mix(in srgb, ${
                b.mean < 0 ? "var(--neg)" : "var(--accent)"
              } ${(Math.sqrt(Math.abs(b.mean) / binMax) * 85).toFixed(0)}%, transparent)`,
            }}
          />
        ))}
      </div>
      <div ref={wrapRef} className="traceread">
        {tokens.map((t, i) => {
          const a = Math.sqrt(Math.abs(t.d) / max);
          return (
            <span
              key={i}
              data-tok={i % binN === 0 ? i : undefined}
              className={`${i === mark ? "divmark" : ""}${CTL.test(t.t) ? " ctl" : ""}` || undefined}
              style={{
                background: `color-mix(in srgb, ${
                  t.d < 0 ? "var(--neg)" : "var(--accent)"
                } ${(a * 80).toFixed(0)}%, transparent)`,
              }}
              title={`token ${i}: Δ ${t.d.toFixed(3)} nats`}
            >
              {t.t}
            </span>
          );
        })}
      </div>
    </>
  );
}

/** The baseline's actual trace text, heat-mapped by per-token Δlogprob. Falls back to the bar sparkline when token ids aren't in the store. */
export function TraceView({
  base,
  ckpt,
  row,
  sample,
  delta,
  mark,
}: {
  base: string;
  ckpt: string;
  row: string;
  sample: number;
  delta: number[];
  mark: number | null;
}) {
  const [toks, setToks] = useState<{ t: string; d: number }[] | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setToks(null);
    setFailed(false);
    api
      .divergenceTrace(base, ckpt, row, sample)
      .then((r) => setToks(r.tokens))
      .catch(() => setFailed(true));
  }, [base, ckpt, row, sample]);

  if (failed) return <Spark delta={delta} mark={mark} w={800} h={72} />;
  if (!toks) return <p className="sub">loading trace…</p>;
  return <Trace tokens={toks} mark={mark} />;
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
}: {
  delta: number[];
  mark: number | null;
  w?: number;
  h?: number;
  n?: number;
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
