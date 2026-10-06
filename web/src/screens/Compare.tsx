import { useEffect, useState } from "react";
import { api, type PairReport, type RowDetail, type RowSample, type Run } from "../api";
import Tip from "../Tip";
import Md from "../Md";
import { firstDivPos, Trace } from "../Trace";
import { runLabel } from "../runLabel";

function hashParams(): URLSearchParams {
  return new URLSearchParams(window.location.hash.split("?")[1] ?? "");
}

export default function Compare() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [mh, setMh] = useState("");
  const [a, setA] = useState(hashParams().get("a") ?? "");
  const [b, setB] = useState(hashParams().get("b") ?? "");
  const [pair, setPair] = useState<PairReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // run records don't always carry the manifest name; resolve it by hash
  const [manifestNames, setManifestNames] = useState<Record<string, string>>({});

  const step = (r: Run) =>
    (r.provenance?.train_step_measured as number) ??
    r.train_step ??
    (r.label === "base" || r.label === "baseline" ? 0 : 1e9);

  // first/last by step within the largest study on the manifest (ties → most
  // recent); a cross-study pair is a deliberate choice, never the default
  const defaultPair = (group: Run[]): [string, string] => {
    const byStudy = new Map<string, Run[]>();
    for (const r of group) {
      const s = r.study ?? "";
      byStudy.set(s, [...(byStudy.get(s) ?? []), r]);
    }
    const created = (rs: Run[]) =>
      Math.max(...rs.map((r) => Date.parse(String(r.created_at ?? "")) || 0));
    const biggest = [...byStudy.values()].sort(
      (x, y) => y.length - x.length || created(y) - created(x),
    )[0] ?? [];
    const sorted = [...biggest].sort((x, y) => step(x) - step(y));
    return [sorted[0]?.run_id ?? "", sorted[sorted.length - 1]?.run_id ?? ""];
  };

  useEffect(() => {
    api
      .manifests()
      .then((ms) => setManifestNames(Object.fromEntries(ms.map((m) => [m.manifest_hash, m.name]))))
      .catch(() => {});
    api.runs().then((rs) => {
      setRuns(rs);
      const groups = new Map<string, Run[]>();
      for (const r of rs) {
        const k = r.manifest_hash ?? "";
        groups.set(k, [...(groups.get(k) ?? []), r]);
      }
      // prefer the manifest the deep-linked base run used, else the biggest group
      const want = rs.find((r) => r.run_id === (hashParams().get("a") ?? a))?.manifest_hash;
      const chosen =
        (want != null && groups.get(want)) ||
        [...groups.values()].sort((x, y) => y.length - x.length)[0];
      if (!chosen) return;
      setMh(chosen[0].manifest_hash ?? "");
      const [first, last] = defaultPair(chosen);
      if (!a) setA(first);
      if (!b) setB(last);
    });
  }, []);

  const pickManifest = (v: string) => {
    setMh(v);
    const [first, last] = defaultPair(runs.filter((r) => (r.manifest_hash ?? "") === v));
    setA(first);
    setB(last);
  };

  const manifestGroups = [...new Map(runs.map((r) => [r.manifest_hash ?? "", r])).entries()];
  const onManifest = runs.filter((r) => (r.manifest_hash ?? "") === mh);
  const studyOf = (id: string) => runs.find((r) => r.run_id === id)?.study ?? null;
  const crossStudy =
    pair && studyOf(a) && studyOf(b) && studyOf(a) !== studyOf(b)
      ? `cross-study comparison: ${studyOf(a)} vs ${studyOf(b)} - same questions, but different training runs/configs`
      : null;
  const warnings = [
    ...(crossStudy ? [crossStudy] : []),
    ...(pair?.comparability?.warnings ?? []).filter((w) => w !== crossStudy),
  ];

  useEffect(() => {
    if (!a || !b || a === b) return;
    setPair(null);
    setErr(null);
    api.compare(a, b).then(setPair).catch((e) => setErr(String(e)));
  }, [a, b]);

  return (
    <>
      <h1>compare</h1>
      <p className="page-sub">two runs graded on the same frozen questions - what changed?</p>
      <div className="row">
        <label>
          eval{" "}
          <Tip text="The frozen question set both runs were graded on. Comparing across evals is meaningless - no paired rows - so the run pickers below only offer runs from this eval.">
            ?
          </Tip>{" "}
          <select value={mh} onChange={(e) => pickManifest(e.target.value)}>
            {manifestGroups.map(([h, r]) => {
              const on = runs.filter((x) => (x.manifest_hash ?? "") === h);
              const studies = [...new Set(on.map((x) => x.study ?? "unlabeled"))].sort();
              return (
                <option key={h} value={h}>
                  {r.manifest ?? manifestNames[h] ?? `manifest ${h.slice(0, 10)}`} ·{" "}
                  {studies.join(", ")} · {on.length} runs
                </option>
              );
            })}
          </select>
        </label>
        <RunPicker label="base" runs={onManifest} value={a} onChange={setA} other={runs.find((r) => r.run_id === b)} />
        <RunPicker label="ckpt" runs={onManifest} value={b} onChange={setB} other={runs.find((r) => r.run_id === a)} />
      </div>
      {pair && (
        <ContextLine pair={pair} runs={runs} />
      )}
      {err && <p className="err">{err}</p>}
      {warnings.length > 0 && (
        <div className="banner warn">
          {warnings.map((w) => (
            <div key={w}>⚠ {w}</div>
          ))}
        </div>
      )}
      {pair && <Report pair={pair} runs={runs} />}
    </>
  );
}

/** What am I comparing? study · manifest · label names, for someone who landed via a deep link. */
function ContextLine({ pair, runs }: { pair: PairReport; runs: Run[] }) {
  const ra = runs.find((r) => r.run_id === pair.run_a);
  const rb = runs.find((r) => r.run_id === pair.run_b);
  const bits = [
    ra?.study ?? rb?.study,
    ra?.manifest ?? ra?.manifest_hash?.slice(0, 10),
    ra?.model ?? rb?.model,
  ].filter(Boolean);
  return (
    <p className="sub" style={{ marginTop: -8 }}>
      {runLabel(ra, pair.run_a)} → {runLabel(rb, pair.run_b)}
      {bits.length ? ` · ${bits.join(" · ")}` : ""}
    </p>
  );
}

function RunPicker({
  label,
  runs,
  value,
  onChange,
  other,
}: {
  label: string;
  runs: Run[];
  value: string;
  onChange: (v: string) => void;
  other?: Run;
}) {
  // group by study so cross-study pairs are a deliberate choice, not the default mess
  const studies = [...new Set(runs.map((r) => r.study ?? "unlabeled"))].sort();
  // mirror stats.comparability's blocks: no shared rows → no report
  const blocked = (r: Run): string | null => {
    if (!other || r.run_id === other.run_id) return null;
    if (other.manifest_hash && r.manifest_hash && other.manifest_hash !== r.manifest_hash)
      return "different manifest";
    const go = other.grader_id as string | undefined;
    const gr = r.grader_id as string | undefined;
    if (go && gr && go !== gr) return "different grader";
    return null;
  };
  return (
    <label>
      {label}{" "}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {studies.map((s) => (
          <optgroup key={s} label={s}>
            {runs
              .filter((r) => (r.study ?? "unlabeled") === s)
              .sort((x, y) => (x.train_step ?? -1) - (y.train_step ?? -1))
              .map((r) => {
                const why = blocked(r);
                return (
                  <option
                    key={r.run_id}
                    value={r.run_id}
                    disabled={!!why}
                    title={why ? `cannot compare: ${why}` : undefined}
                  >
                    {runLabel(r)} · k={r.k ?? "?"} · {r.run_id.slice(0, 8)}
                    {why ? ` (${why})` : ""}
                  </option>
                );
              })}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

function Report({ pair, runs }: { pair: PairReport; runs: Run[] }) {
  const ag = pair.agreement;
  const labelA = runLabel(runs.find((r) => r.run_id === pair.run_a), pair.run_a);
  const labelB = runLabel(runs.find((r) => r.run_id === pair.run_b), pair.run_b);
  // ?row= deep link (from the divergence screen) opens that row's panel
  const [open, setOpen] = useState<string | null>(() => {
    const r = hashParams().get("row");
    return r && (pair.cells ?? []).some((c) => c.row_id === r) ? r : null;
  });
  useEffect(() => {
    if (open) document.getElementById("rowpanel")?.scrollIntoView({ block: "start" });
    // mount-only: scroll once when a deep link opens a row
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // failure_kind -> highlighted chip rows; null = no filter
  const [filter, setFilter] = useState<string | null>(null);
  const [qFilter, setQFilter] = useState("");
  const [onlyChanged, setOnlyChanged] = useState(false);
  const [copied, setCopied] = useState(false);
  const failKinds = [
    ...new Set([...Object.keys(pair.failures.a), ...Object.keys(pair.failures.b)]),
  ].sort();
  const passK = Object.keys(pair.passn ?? {}).sort((x, y) => Number(y) - Number(x))[0];
  const passKE = passK ? pair.passn?.[passK] : undefined;
  const filterable = !!Object.keys(pair.failure_rows_b ?? {}).length;
  return (
    <>
      <div className="cards">
        <div className="card">
          <div className="big">
            {pct(pair.acc_a)} → {pct(pair.acc_b)}
          </div>
          <div className="sub">
            <Tip text="Paired per-question accuracy change (b − a), with a bootstrap 95% CI over the shared eval questions. Pairing removes question-difficulty noise; a CI that crosses 0 is inconclusive.">
              accuracy · Δ {pair.delta >= 0 ? "+" : ""}
              {pair.delta.toFixed(3)} [{pair.delta_ci[0].toFixed(3)},{" "}
              {pair.delta_ci[1].toFixed(3)}]
            </Tip>
            {" · "}
            {pair.n_pairs} paired rows
          </div>
        </div>
        {passKE && (
          <div className="card">
            <div className="big">
              {pct(passKE.a)} → {pct(passKE.b)}
            </div>
            <div className="sub">
              <Tip text="at least one of k samples correct per question">
                pass@{passK} · Δ {passKE.delta >= 0 ? "+" : ""}
                {passKE.delta.toFixed(3)} [{passKE.delta_ci[0].toFixed(3)},{" "}
                {passKE.delta_ci[1].toFixed(3)}]
              </Tip>
            </div>
          </div>
        )}
        <div className="card">
          <div className="big">{pair.flips.length}</div>
          <div className="sub">
            <Tip text="Questions whose majority outcome changed between the two runs - regressions (was right, now wrong) vs gains.">
              questions changed outcome
            </Tip>
            {" · "}
            {pair.flips.filter((f) => f.kind === "regression").length} regressions ·{" "}
            {pair.flips.filter((f) => f.kind === "gain").length} gains
          </div>
        </div>
        <div className="card">
          <div className="big">
            {pct(pair.truncation_rate_a)} → {pct(pair.truncation_rate_b)}
          </div>
          <div className="sub">
            <Tip text="Share of responses that ran into the max-token cap mid-answer - a shape-of-failure signal, not a correctness one.">
              hit token cap
            </Tip>
          </div>
        </div>
        <div className="card">
          <div className="big">
            <Tip text="Average response length in generated tokens. A big paired shift means the output regime changed - rambling or collapsing - not just accuracy.">
              {pair.tokens.a.mean.toFixed(0)} → {pair.tokens.b.mean.toFixed(0)}
            </Tip>
          </div>
          <div className="sub">
            avg response length · Δ {pair.delta_tokens >= 0 ? "+" : ""}
            {pair.delta_tokens.toFixed(0)} [{pair.delta_tokens_ci[0].toFixed(0)},{" "}
            {pair.delta_tokens_ci[1].toFixed(0)}]
          </div>
        </div>
        <div className="card">
          <div className="big">
            ${pair.cost_a.toFixed(2)} vs ${pair.cost_b.toFixed(2)}
          </div>
          <div className="sub">estimated cost</div>
        </div>
      </div>

      <PairRead pair={pair} />

      <h2>
        <Tip text="How each shared question landed under both runs: the off-diagonals are the flips.">
          per-question outcomes
        </Tip>
      </h2>
      <table className="agreement">
        <tbody>
          <tr>
            <td>both right <b>{ag.both_right}</b></td>
            <td>only {labelA} right <b>{ag.a_only}</b></td>
          </tr>
          <tr>
            <td>only {labelB} right <b>{ag.b_only}</b></td>
            <td>both wrong <b>{ag.both_wrong}</b></td>
          </tr>
        </tbody>
      </table>

      {failKinds.length > 0 && (
        <>
          <h2>
            <Tip text="Why the wrong samples were wrong: truncated at the token cap, no parseable answer, wrong answer, etc. This is the shape of the failure - the difference between 'degenerate outputs' and 'just wrong'. Click a kind to light up its rows in the grid below.">
              why samples failed
            </Tip>
          </h2>
          <table>
            <thead>
              <tr>
                <th>kind</th>
                <th>{labelA}</th>
                <th>{labelB}</th>
                <th>Δ</th>
              </tr>
            </thead>
            <tbody>
              {failKinds.map((k) => {
                const na = pair.failures.a[k] ?? 0;
                const nb = pair.failures.b[k] ?? 0;
                return (
                  <tr
                    key={k}
                    className={
                      filterable ? `fliprow${filter === k ? " fk-on" : ""}` : undefined
                    }
                    title={
                      filterable ? "click to light up this kind's rows in the grid below" : undefined
                    }
                    onClick={
                      filterable ? () => setFilter(filter === k ? null : k) : undefined
                    }
                  >
                    <td>{k.replace(/_/g, " ")}</td>
                    <td>{na}</td>
                    <td>{nb}</td>
                    <td className={nb - na > 0 ? "neg" : nb - na < 0 ? "pos" : ""}>
                      {nb - na >= 0 ? "+" : ""}
                      {nb - na}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      {pair.cells && pair.cells.length > 0 && (
        <>
          <h2>
            <Tip text="One chip per question in the eval. Fill = the row's accuracy shift (green improved, red regressed, muted unchanged); a colored border means its majority status actually flipped. Click a chip for the row's outputs; click a failure kind above to light up its rows.">
              every question
            </Tip>
          </h2>
          <div className="gridcontrols">
            <input
              className="qfilter"
              placeholder="filter questions…"
              value={qFilter}
              onChange={(e) => setQFilter(e.target.value)}
            />
            <label className="sub">
              <input
                type="checkbox"
                checked={onlyChanged}
                onChange={(e) => setOnlyChanged(e.target.checked)}
              />{" "}
              only changed
            </label>
          </div>
          <div className="chipgrid">
            {pair.cells.map((c) => {
              const hit =
                filter != null &&
                (pair.failure_rows_b?.[filter] ?? []).includes(c.row_id);
              const qHit =
                qFilter.trim() === "" ||
                (c.q ?? c.row_id).toLowerCase().includes(qFilter.trim().toLowerCase());
              const dp =
                c.p_a != null && c.p_b != null ? c.p_b - c.p_a : null;
              const changedHit =
                !onlyChanged ||
                c.cell === "a_only" ||
                c.cell === "b_only" ||
                (dp != null && dp !== 0);
              const hard = c.cell === "a_only" || c.cell === "b_only";
              const dim =
                (filter != null && !hit) || !qHit || !changedHit;
              const bg =
                c.cell === "excluded" || dp == null || (dp === 0 && c.cell === "both_wrong")
                  ? undefined
                  : dp === 0
                    ? "color-mix(in srgb, var(--pos) 25%, transparent)"
                    : `color-mix(in srgb, ${
                        dp > 0 ? "var(--pos)" : "var(--neg)"
                      } ${35 + Math.abs(dp) * 55}%, transparent)`;
              return (
                <div
                  key={c.row_id}
                  className={`chip ${c.cell === "excluded" ? "excluded" : ""}${
                    dp === 0 && c.cell === "both_wrong" ? " flatwrong" : ""
                  }${
                    hard ? (c.cell === "b_only" ? " hard_up" : " hard_dn") : ""
                  }${dim ? " dim" : ""}${hit ? " hit" : ""}${
                    open === c.row_id ? " open" : ""
                  }`}
                  style={bg ? { background: bg } : undefined}
                  title={`${c.q ? c.q + "\n" : ""}${c.row_id} · ${c.p_a == null ? "-" : c.p_a.toFixed(2)} → ${c.p_b == null ? "-" : c.p_b.toFixed(2)}${hard ? " · status flip" : dp && dp !== 0 ? " · reliability shift" : ""}`}
                  onClick={() => setOpen(open === c.row_id ? null : c.row_id)}
                />
              );
            })}
          </div>
          {(() => {
            const flipsN = pair.cells!.filter(
              (c) => c.cell === "a_only" || c.cell === "b_only",
            ).length;
            const softN = pair.cells!.filter(
              (c) =>
                c.p_a != null && c.p_b != null && c.p_b !== c.p_a &&
                c.cell !== "a_only" && c.cell !== "b_only",
            ).length;
            return (
              <p className="sub">
                {flipsN} status flip{flipsN === 1 ? "" : "s"} · {softN} row
                {softN === 1 ? "" : "s"} shifted reliability without flipping
                {flipsN === 0 && softN === 0
                  ? " - identical row-level outcomes"
                  : ""}
              </p>
            );
          })()}
          {filter != null && (
            <p className="sub">
              filtered by {filter.replace(/_/g, " ")} (
              {(pair.failure_rows_b?.[filter] ?? []).length} rows) - click the kind
              again to clear
            </p>
          )}
          <div className="legend">
            <span><span className="sw up" />improved</span>
            <span><span className="sw dn" />regressed</span>
            <span><span className="sw flat" />unchanged · right</span>
            <span><span className="sw flatwrong" />unchanged · wrong</span>
            <span><span className="sw excluded" />excluded</span>
            <span>border = status flip</span>
          </div>
        </>
      )}

      {open && (
        <div className="rowpanel" id="rowpanel">
          <RowPanel pair={pair} row={open} labelA={labelA} labelB={labelB} />
        </div>
      )}

      {pair.flips.length > 0 && (
        <>
          <h2>
            questions that flipped{" "}
            <button
              className="theme-btn"
              onClick={() => {
                navigator.clipboard.writeText(
                  JSON.stringify(
                    pair.flips.map((f) => ({
                      row_id: f.row_id, question: f.q, p_a: f.p_a,
                      p_b: f.p_b, kind: f.kind, hard: f.hard,
                    })),
                    null,
                    2,
                  ),
                );
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }}
            >
              {copied ? "copied" : "copy as json"}
            </button>
          </h2>
          <table>
            <thead>
              <tr>
                <th>question</th>
                <th><Tip text="Base model's pass rate on this question across its k samples.">p(base)</Tip></th>
                <th><Tip text="Checkpoint's pass rate on this question across its k samples.">p(ckpt)</Tip></th>
                <th><Tip text="regression = right→wrong, gain = wrong→right. (hard) = every sample flipped, not just a marginal one - the strongest evidence of a real change.">kind</Tip></th>
              </tr>
            </thead>
            <tbody>
              {pair.flips.map((f) => (
                <tr
                  key={f.row_id}
                  className={`${f.kind} fliprow${open === f.row_id ? " fk-on" : ""}`}
                  onClick={() => setOpen(open === f.row_id ? null : f.row_id)}
                >
                  <td title={f.q ?? f.row_id} style={{ maxWidth: 420 }}>
                    {f.q ? <div className="qcell-md"><Md text={f.q} /></div> : <span className="mono">{f.row_id}</span>}
                  </td>
                  <td>{f.p_a.toFixed(2)}</td>
                  <td>{f.p_b.toFixed(2)}</td>
                  <td>
                    {f.kind}
                    {f.hard ? " (hard)" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {pair.excluded.length > 0 && (
        <>
          <h2>excluded ({pair.excluded.length})</h2>
          <table>
            <tbody>
              {pair.excluded.map((e) => (
                <tr key={e.row_id}>
                  <td className="mono">{e.row_id}</td>
                  <td>{e.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </>
  );
}

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}

/** Plain-English summary of the pair report - the numbers above, said once. */
function PairRead({ pair }: { pair: PairReport }) {
  const [lo, hi] = pair.delta_ci;
  const bullets: string[] = [
    `accuracy ${pct(pair.acc_a)} → ${pct(pair.acc_b)}; Δ ${pair.delta >= 0 ? "+" : ""}${pair.delta.toFixed(3)} with 95% CI [${lo.toFixed(3)}, ${hi.toFixed(3)}] - ${
      hi < 0 || lo > 0
        ? "the CI excludes zero: this is a real change, not sampling noise"
        : "the CI includes zero: inconclusive at this sample size"
    }`,
  ];

  const regs = pair.flips.filter((f) => f.kind === "regression").length;
  const gains = pair.flips.filter((f) => f.kind === "gain").length;
  if (pair.flips.length >= 3) {
    bullets.push(
      `${regs} regressions vs ${gains} gains - ${
        Math.min(regs, gains) <= 0.15 * Math.max(regs, gains)
          ? "one-directional, which sampling noise doesn't produce"
          : "roughly symmetric, consistent with noise"
      }`,
    );
  }

  const ta = pair.tokens.a.mean;
  const tb = pair.tokens.b.mean;
  if (Math.max(ta, tb) / Math.max(1, Math.min(ta, tb)) > 3) {
    bullets.push(
      `mean response length ${ta.toFixed(0)} → ${tb.toFixed(0)} tokens - the output regime changed, not just the accuracy`,
    );
  }

  if (Math.abs(pair.truncation_rate_b - pair.truncation_rate_a) > 0.2) {
    bullets.push(
      `truncation ${pct(pair.truncation_rate_a)} → ${pct(pair.truncation_rate_b)} - ${
        pair.truncation_rate_b > pair.truncation_rate_a
          ? "the checkpoint rambles into the token cap"
          : "the checkpoint stopped hitting the token cap"
      }`,
    );
  }

  const kinds = new Set([...Object.keys(pair.failures.a), ...Object.keys(pair.failures.b)]);
  let topKind = "";
  let topDelta = 0;
  for (const k of kinds) {
    const d = (pair.failures.b[k] ?? 0) - (pair.failures.a[k] ?? 0);
    if (d > topDelta) {
      topDelta = d;
      topKind = k;
    }
  }
  if (topDelta >= 3) {
    bullets.push(`most new failures are "${topKind.replace(/_/g, " ")}" (+${topDelta})`);
  }

  return (
    <div className="card read">
      <div className="mini-label">
        <Tip text="Plain sentences generated deterministically from the numbers above - thresholds and templates, no LLM involved.">
          takeaways
        </Tip>
      </div>
      {bullets.map((b) => (
        <p key={b}>{b}</p>
      ))}
    </div>
  );
}

/** Expanded flip row: the question, then both runs' raw samples side by side. */
function RowPanel({
  pair,
  row,
  labelA,
  labelB,
}: {
  pair: PairReport;
  row: string;
  labelA: string;
  labelB: string;
}) {
  const [d, setD] = useState<RowDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [toks, setToks] = useState<{ t: string; d: number }[] | null>(null);
  // different renderers → no shared vocabulary, so per-token views are meaningless
  const tokenViews = pair.comparability?.token_views !== false;
  useEffect(() => {
    setD(null);
    setErr(null);
    api.compareRow(pair.run_a, pair.run_b, row).then(setD).catch((e) => setErr(String(e)));
  }, [pair.run_a, pair.run_b, row]);
  useEffect(() => {
    setToks(null);
    if (!tokenViews) return;
    // no divergence computed for this pair → the section just stays out
    api
      .divergenceTrace(pair.run_a, pair.run_b, row, 0)
      .then((r) => setToks(r.tokens))
      .catch(() => {});
  }, [pair.run_a, pair.run_b, row, tokenViews]);

  if (err) return <p className="err">{err}</p>;
  if (!d) return <p className="sub">loading…</p>;

  const divPos = toks ? firstDivPos(toks.map((t) => t.d)) : null;
  const q = [...d.question].reverse().find((m) => m.role === "user")?.content ?? "";
  return (
    <div>
      <div className="qtext"><Md text={q} /></div>
      <div className="sub">expected answer: {d.answer}</div>
      <div className="outputs">
        {/* cells emit in row order so sample i of each run shares a grid row */}
        <div className="mini-label">
          {labelA}
          {d.k_a != null ? ` · ${d.k_a} sample${d.k_a === 1 ? "" : "s"}` : ""}
        </div>
        <div className="mini-label">
          {labelB}
          {d.k_b != null ? ` · ${d.k_b} sample${d.k_b === 1 ? "" : "s"}` : ""}
        </div>
        {Array.from({ length: Math.max(d.a.length, d.b.length) }, (_, i) =>
          [d.a[i], d.b[i]].map((s, side) =>
            s ? (
              <SampleView key={`${i}${side}`} s={s} />
            ) : (
              <div key={`${i}${side}`} className="sub" style={{ opacity: 0.5 }}>
                (no sample {i})
              </div>
            ),
          ),
        )}
      </div>
      {toks && (
        <div style={{ marginTop: 12 }}>
          <div className="mini-label">
            where ckpt diverges on base's trace (sample 0)
          </div>
          <Trace tokens={toks} mark={divPos} />
          <div className="sub" style={{ marginTop: 6 }}>
            {toks.length.toLocaleString()} tokens · Σ{" "}
            {toks.reduce((a, t) => a + t.d, 0).toFixed(1)} nats
            {divPos != null
              ? ` · first divergence at token ${divPos.toLocaleString()}`
              : ""}
          </div>
        </div>
      )}
      {tokenViews ? (
        <div className="sub">
          <a
            href={`#/divergence?base=${pair.run_a}&ckpt=${pair.run_b}&row=${encodeURIComponent(row)}&sample=0`}
          >
            view divergence trace →
          </a>
        </div>
      ) : (
        <div className="sub">per-token divergence unavailable: different renderers</div>
      )}
    </div>
  );
}

const TAIL = 1200;

/** Last ~TAIL chars, cut at a paragraph break so a math block isn't split. */
function tail(text: string): string {
  const from = text.length - TAIL;
  const brk = text.indexOf("\n\n", from);
  return text.slice(brk >= 0 && brk - from <= 600 ? brk + 2 : from);
}

function clamp(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

function SampleView({ s }: { s: RowSample }) {
  const [full, setFull] = useState(false);
  const [think, setThink] = useState(false);
  const ok = s.verdict === 1;
  const text = s.text_clean ?? s.text.replace(/<\|[^|>]+\|>/g, "").trim();
  const empty = text.length === 0;
  const long = text.length > TAIL;
  const shown = long && !full ? tail(text) : text;
  const thinking = s.thinking ?? "";
  if (empty) {
    return (
      <div style={{ marginBottom: 10 }}>
        <div>
          <span className="badge neg">✗ empty</span>{" "}
          <span className="sub">
            {s.gen_tokens.toLocaleString()} token{s.gen_tokens === 1 ? "" : "s"} · empty
            response - the model emitted only the end-of-message token
          </span>
        </div>
        <div className="togglerow">
          <button className="theme-btn" style={{ visibility: "hidden" }}>&nbsp;</button>
        </div>
        <div className="outtext"><span className="muted">(no output)</span></div>
      </div>
    );
  }
  return (
    <div style={{ marginBottom: 10 }}>
      <div>
        <span className={`badge ${ok ? "pos" : "neg"}`}>
          {ok ? "✓ correct" : `✗ ${(s.failure_kind ?? "wrong").replace(/_/g, " ")}`}
        </span>{" "}
        <span className="sub">
          <Tip text="tokens the model generated for this sample, thinking included. The text shown is only the final message; the thinking is below it.">
            {s.gen_tokens.toLocaleString()} tokens incl. thinking
          </Tip>
          {" · "}
          <Tip text="stop reason: 'stop' = ended naturally at a message boundary, 'length' = hit the max-token cap mid-thought (truncated)">
            {s.stop_reason === "length" ? "truncated at cap" : "ended naturally"}
          </Tip>
          {" · "}
          <Tip text="the final answer the grader extracted from the text, shown literally - the verdict compares it to the expected answer numerically, so '050' == '50'">
            extracted: <span className="ext" title={s.extracted ?? undefined}>{clamp(s.extracted ?? "-", 40)}</span>
          </Tip>
          {s.grade_note && (
            <>
              {" · "}
              <Tip text="what the grader reported - for custom graders and judges this is the reason the verdict was given">
                <span className="muted" title={s.grade_note}>{clamp(s.grade_note, 50)}</span>
              </Tip>
            </>
          )}
        </span>
      </div>
      {/* reserved row: every cell gets one toggles line so paired boxes
          stay aligned whether or not buttons apply */}
      <div className="togglerow">
        <button
          className="theme-btn"
          style={{ visibility: long ? "visible" : "hidden" }}
          onClick={() => setFull(!full)}
        >
          {full ? "show tail" : `show full (${text.length.toLocaleString()} chars)`}
        </button>
        <button
          className="theme-btn"
          style={{ visibility: thinking.length > 0 ? "visible" : "hidden" }}
          onClick={() => setThink(!think)}
        >
          {think ? "hide thinking" : `thinking (${thinking.length.toLocaleString()} chars)`}
        </button>
      </div>
      <div className="outtext"><Md text={shown} /></div>
      {think && thinking.length > 0 && (
        <div className="outtext thinking"><Md text={thinking} /></div>
      )}
    </div>
  );
}
