import { useEffect, useRef, useState } from "react";
import { api, type RolloutEvent, type RolloutSummary, type RolloutText, type RolloutTexts } from "../api";
import Md from "../Md";
import Tip from "../Tip";

/** Rollout metrics and sampled text are separate evidence; never pair their rows. */
export default function Rollouts({ study }: { study: string }) {
  const [data, setData] = useState<RolloutSummary | null>(null);
  const [error, setError] = useState("");
  const [split, setSplit] = useState("train");
  useEffect(() => {
    let active = true;
    api.rollouts(study).then((d) => {
      if (!active) return;
      setData(d);
      setSplit(d.splits.includes("train") ? "train" : d.splits[0] ?? "");
    }).catch((e) => {
      if (active && !String(e).includes("404 ")) setError(String(e));
    });
    return () => { active = false; };
  }, [study]);
  if (error) return <p className="err">{error}</p>;
  if (!data || !split) return null;
  return (
    <section className="card rollout-section" aria-label="training rollouts">
      <div className="row">
        <h2>training rollouts</h2>
        <Tip text="Metrics cover the imported rollout summaries. Transcripts are a logged sample with no reliable link to summary IDs.">
          <span className="sub">about this data</span>
        </Tip>
        {data.splits.length > 1 && <label>split{" "}
          <select aria-label="split" value={split} onChange={(e) => setSplit(e.target.value)}>
            {data.splits.map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>}
      </div>
      <SplitView key={split} study={study} split={split} data={data} />
    </section>
  );
}

function SplitView({ study, split, data }: { study: string; split: string; data: RolloutSummary }) {
  const points = data.series[split];
  const coverage = data.coverage[split];
  const iterations = points.filter((p) => p.n_texts > 0).map((p) => p.iteration);
  const [iteration, setIteration] = useState(iterations.at(-1) ?? points[points.length - 1].iteration);
  const [compare, setCompare] = useState("");
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState("as_logged");
  const components = [...new Set(points.flatMap((p) => Object.keys(p.components)))].sort();
  const curves = [
    { name: "total reward", points: points.map((p) => ({ x: p.iteration, y: p.reward })) },
    ...components.map((name) => ({
      name, points: points.map((p) => ({ x: p.iteration, y: p.components[name] ?? null })),
    })),
  ];
  const fractions = curves.filter((c) => c.points.every((p) => p.y == null || (p.y >= 0 && p.y <= 1)));
  const other = curves.filter((c) => !fractions.includes(c));
  const [otherMetric, setOtherMetric] = useState(other[0]?.name ?? "");
  const events = data.events[split];
  const selected = other.find((c) => c.name === otherMetric);
  return (
    <>
      <p className="sub">
        {coverage.rollouts.toLocaleString()} rollouts over {points.length} iterations ·{" "}
        {coverage.texts.toLocaleString()} logged transcripts ({(100 * coverage.texts / coverage.rollouts).toFixed(1)}%)
      </p>
      <p className="sub">
        One vote per episode; response tokens sum all steps. Components use final-step metrics where recorded.
        Questions can differ between iterations; these trends are not paired eval comparisons.
      </p>
      {fractions.length > 0 && <RolloutChart title="reward and logged metrics (0–1)"
        curves={fractions} fraction events={events} onSelect={setIteration} />}
      {other.length > 0 && <>
        <label>other metric{" "}
          <select aria-label="other metric" value={otherMetric} onChange={(e) => setOtherMetric(e.target.value)}>
            {other.map((c) => <option key={c.name}>{c.name}</option>)}
          </select>
        </label>
        {selected && <RolloutChart title={selected.name} curves={[selected]} onSelect={setIteration} />}
      </>}
      {events.length > 0 && <div className="rollout-events">
        <p className="sub">Threshold hints: trailing 3-iteration means; not significance tests.</p>
        <ul>{events.map((e) => (
          <li key={`${e.component}-${e.kind}`}>
            <button type="button" onClick={() => setIteration(e.iteration)}>
              {e.component} {e.kind === "saturated" ? "reached ≥0.98" : "fell ≥0.10 from its prior peak"}
              {" "}at iteration {e.iteration} ({e.from.toFixed(2)} → {e.to.toFixed(2)})
            </button>
          </li>
        ))}</ul>
      </div>}
      <RolloutChart title="mean response tokens per episode"
        curves={[{ name: "tokens", points: points.map((p) => ({ x: p.iteration, y: p.tokens })) }]}
        onSelect={setIteration} />
      <details className="rollout-data">
        <summary>iteration values and coverage</summary>
        <div className="rollout-table"><table>
          <thead><tr><th>iteration</th><th>rollouts</th><th>transcripts</th><th>tokens</th>
            <th>reward</th>{components.map((k) => <th key={k}>{k} (n)</th>)}</tr></thead>
          <tbody>{points.map((p) => <tr key={p.iteration}>
            <td><button type="button" onClick={() => setIteration(p.iteration)}>{p.iteration}</button></td>
            <td>{p.n}</td><td>{p.n_texts}</td><td>{p.tokens.toFixed(0)}</td><td>{p.reward?.toFixed(3) ?? "—"}</td>
            {components.map((k) => <td key={k}>{p.components[k]?.toFixed(3) ?? "—"} ({p.component_n[k] ?? 0})</td>)}
          </tr>)}</tbody>
        </table></div>
      </details>
      <h2>logged transcripts</h2>
      <p className="sub">These are a sample, unlinked to the rollout IDs above.
        Side-by-side lists contain different questions, not matched before/after attempts.
        Length is in characters, not tokens.</p>
      {!coverage.texts ? <p className="sub">No transcripts were logged for this split.</p> : <>
        <div className="row rollout-controls">
          <label>iteration{" "}
            <select aria-label="iteration" value={iteration} onChange={(e) => setIteration(Number(e.target.value))}>
              {points.map((p) => <option key={p.iteration} value={p.iteration}>
                {p.iteration}{p.n_texts ? "" : " (no text)"}
              </option>)}
            </select>
          </label>
          <label>compare with iteration{" "}
            <select aria-label="compare with iteration" value={compare} onChange={(e) => setCompare(e.target.value)}>
              <option value="">none</option>
              {iterations.map((i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </label>
          <label>sort{" "}
            <select aria-label="sort" value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="as_logged">as logged</option>
              <option value="short">shortest (chars)</option>
              <option value="long">longest (chars)</option>
            </select>
          </label>
        </div>
        <div className="rollout-filters" aria-label="transcript filters">
          <button type="button" aria-pressed={filter === ""} onClick={() => setFilter("")}>all</button>
          {(data.text_fields[split] ?? []).flatMap((field) => ["False", "True"].map((value) => {
            const key = JSON.stringify([field, value]);
            return <button type="button" key={key} aria-pressed={filter === key}
              onClick={() => setFilter(key)}>{field}: {value}</button>;
          }))}
        </div>
        <div className={compare === "" ? "rollout-columns" : "rollout-columns paired"}>
          <TranscriptList key={`${study}-${split}-${iteration}-${filter}-${sort}`}
            study={study} split={split} iteration={iteration} filter={filter} sort={sort} />
          {compare !== "" && <TranscriptList key={`${study}-${split}-${compare}-${filter}-${sort}-compare`}
            study={study} split={split} iteration={Number(compare)} filter={filter} sort={sort} />}
        </div>
      </>}
    </>
  );
}

type Curve = { name: string; points: { x: number; y: number | null }[] };
const COLORS = ["var(--accent)", "var(--pos)", "var(--warn)", "var(--accent2)", "var(--neg)", "var(--muted)"];

function RolloutChart({ title, curves, fraction = false, events = [], onSelect }: {
  title: string; curves: Curve[]; fraction?: boolean; events?: RolloutEvent[]; onSelect: (i: number) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(775);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(250, entry.contentRect.width)));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const all = curves.flatMap((c) => c.points);
  const values = all.flatMap((p) => p.y == null ? [] : [p.y]);
  if (!values.length) return null;
  const x0 = Math.min(...all.map((p) => p.x)), x1 = Math.max(...all.map((p) => p.x));
  const y0 = fraction ? 0 : Math.min(0, ...values), y1 = fraction ? 1 : Math.max(1, ...values);
  const x = (v: number) => 55 + (v - x0) / (x1 - x0 || 1) * (width - 80);
  const y = (v: number) => 140 - (v - y0) / (y1 - y0) * 120;
  return (
    <div className="rollout-chart" ref={container}>
      <div className="mini-label">{title} · click a point to inspect its iteration</div>
      <div className="rollout-legend">{curves.map((c, i) => <span key={c.name}>
        <i style={{ background: COLORS[i % COLORS.length] }} />{c.name}
      </span>)}</div>
      <svg viewBox={`0 0 ${width} 175`} role="img" aria-label={title}>
        {[y0, (y0 + y1) / 2, y1].map((v) => <g key={v}>
          <line x1={55} x2={width - 25} y1={y(v)} y2={y(v)} stroke="var(--border)" />
          <text x={47} y={y(v) + 4} textAnchor="end">{fraction ? v.toFixed(1) : v.toFixed(0)}</text>
        </g>)}
        {[...new Set([x0, Math.round((x0 + x1) / 2), x1])].map((v) =>
          <text key={v} x={x(v)} y={161} textAnchor="middle">{v}</text>)}
        {curves.map((c, i) => {
          let move = true;
          const path = c.points.map((p) => {
            if (p.y == null) { move = true; return ""; }
            const segment = `${move ? "M" : "L"}${x(p.x)},${y(p.y)}`;
            move = false;
            return segment;
          }).join(" ");
          const color = COLORS[i % COLORS.length];
          return <g key={c.name}>
            <path d={path} fill="none" stroke={color} strokeWidth={2} />
            {c.points.map((p) => {
              if (p.y == null) return null;
              const event = events.find((e) => e.component === c.name && e.iteration === p.x);
              return <circle key={p.x} cx={x(p.x)} cy={y(p.y)} r={event ? 5 : 3}
                fill={color} stroke={event ? "var(--text)" : "none"}
                className="rollout-point" onClick={() => onSelect(p.x)}>
                <title>{c.name}: {p.y.toFixed(3)} · iteration {p.x}</title>
              </circle>;
            })}
          </g>;
        })}
      </svg>
    </div>
  );
}

function TranscriptList({ study, split, iteration, filter, sort }: {
  study: string; split: string; iteration: number; filter: string; sort: string;
}) {
  const [data, setData] = useState<RolloutTexts | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const params = new URLSearchParams({ split, iteration: String(iteration), sort, limit: "50" });
    if (filter) {
      const [field, value] = JSON.parse(filter);
      params.set("field", field); params.set("value", value);
    }
    api.rolloutTexts(study, params).then((d) => {
      if (active) setData(d);
    }).catch((e) => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [study, split, iteration, filter, sort]);
  return (
    <div className="rollout-list">
      <h3>iteration {iteration}</h3>
      {error ? <p className="err">{error}</p> : !data ? <p className="sub">loading transcripts…</p> : <>
        <p className="sub">{data.total} matching · showing {data.rows.length}</p>
        {!data.rows.length && <p className="sub">No logged transcripts match this selection.</p>}
        {data.rows.map((r) => <Transcript key={`${r.group_pos}-${r.traj_pos}`} row={r} />)}
      </>}
    </div>
  );
}

function Transcript({ row }: { row: RolloutText }) {
  const [full, setFull] = useState(false);
  return (
    <article className="rollout-transcript">
      <div className="sub">log position {row.group_pos + 1}.{row.traj_pos + 1} · {row.chars.toLocaleString()} chars</div>
      <details className="rollout-question">
        <summary title={row.question}>{row.question || "(no question text logged)"}</summary>
        <Md text={row.question} />
      </details>
      <div className="rollout-fields">{Object.entries(row.reward_fields).map(([k, v]) =>
        <span key={k}>{k}: {v}</span>)}</div>
      {row.thinking ? <>
        <div className="mini-label">thinking</div>
        <pre>{full || row.thinking.length <= 240 ? row.thinking : `${row.thinking.slice(0, 240)}…`}</pre>
        {row.thinking.length > 240 && <button type="button" onClick={() => setFull(!full)}>
          {full ? "show less" : "show full thinking"}
        </button>}
      </> : <p className="sub">no thinking text logged</p>}
      <div className="mini-label">answer</div>
      {row.answer ? <Md text={row.answer} /> : <p className="sub">no answer text logged</p>}
    </article>
  );
}
