import { useEffect, useState } from "react";
import { api } from "./api";

// same cumulative-nats threshold the backend's divergence_pos uses (diverge.TAU)
const DIVERGE_TAU = 5.0;

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
  const max = Math.max(1e-9, ...tokens.map((t) => Math.abs(t.d)));
  return (
    <>
      <div className="sub" style={{ marginBottom: 6 }}>
        baseline trace colored by Δlogprob per token —{" "}
        <span style={{ color: "var(--neg)" }}>red</span> = ckpt less confident,{" "}
        <span style={{ color: "var(--accent)" }}>blue</span> = more confident
        {mark != null ? " · amber edge = first divergence" : ""}
      </div>
      <div className="traceread">
        {tokens.map((t, i) => {
          const a = Math.sqrt(Math.abs(t.d) / max);
          return (
            <span
              key={i}
              className={i === mark ? "divmark" : undefined}
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

/** Delta sparkline: red = ckpt less confident than base, blue = more. */
export function Spark({
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
