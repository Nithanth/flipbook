import { useEffect, useState } from "react";
import { api, type Run } from "../api";
import { runLabel } from "../runLabel";

export default function Runs() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // manifest_hash -> display name; run records carry only the hash
  const [mNames, setMNames] = useState<Record<string, string>>({});
  const [sel, setSel] = useState<string[]>([]);
  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setErr(String(e)));
    api.manifests().then((ms) =>
      setMNames(Object.fromEntries(ms.map((m) => [m.manifest_hash, m.name]))),
    );
  }, []);

  if (err) return <p className="err">{err}</p>;
  if (!runs) return <p>loading…</p>;

  const studies = [...new Set(runs.map((r) => r.study ?? "ungrouped"))].sort(
    (a, b) =>
      runs.filter((r) => (r.study ?? "ungrouped") === b).length -
      runs.filter((r) => (r.study ?? "ungrouped") === a).length,
  );

  const selRuns = sel.map((id) => runs.find((r) => r.run_id === id)!);
  const comparable =
    selRuns.length === 2 &&
    selRuns[0].manifest_hash != null &&
    selRuns[0].manifest_hash === selRuns[1].manifest_hash;
  const toggle = (id: string) =>
    setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-2)));

  return (
    <>
      <h1>evals</h1>
      <p className="page-sub">
        every model artifact evaluated on a manifest, grouped by study
        {sel.length === 2 &&
          (comparable ? (
            <>
              {" - "}
              <a href={`#/compare?a=${sel[0]}&b=${sel[1]}`}>compare the selected pair →</a>
            </>
          ) : (
            " - selected runs were graded on different manifests; they can't be paired"
          ))}
        {sel.length === 1 && " - select one more run on the same eval to compare"}
      </p>
      {studies.map((s) => (
        <section key={s}>
          <h2>
            {s}
            {s !== "ungrouped" && (
              <>
                {" "}
                <a href={`#/study?name=${s}`} className="sub">→ study</a>
              </>
            )}
          </h2>
          <table>
            <thead>
              <tr>
                <th></th>
                <th>label</th>
                <th>acc</th>
                <th>model</th>
                <th>effort</th>
                <th>k</th>
                <th>manifest</th>
                <th>created</th>
                <th>run</th>
              </tr>
            </thead>
            <tbody>
              {runs
                .filter((r) => (r.study ?? "ungrouped") === s)
                .sort((x, y) => (x.train_step ?? -1) - (y.train_step ?? -1))
                .map((r) => (
                  <tr key={r.run_id}>
                    <td>
                      <input
                        type="checkbox"
                        checked={sel.includes(r.run_id)}
                        onChange={() => toggle(r.run_id)}
                      />
                    </td>
                    <td>{runLabel(r)}</td>
                    <td className="mono">
                      {r.acc != null ? `${((r.acc as number) * 100).toFixed(0)}%` : "-"}
                    </td>
                    <td className="mono" title={r.model_id as string | undefined}>
                      {shortModel((r.model_id ?? r.model) as string | undefined)}
                    </td>
                    <td>{r.effort ?? ""}</td>
                    <td>{r.k ?? ""}</td>
                    <td title={r.manifest_hash}>
                      {mNames[r.manifest_hash ?? ""] ??
                        (r.manifest_hash ?? "").slice(0, 10)}
                    </td>
                    <td className="sub">
                      {(r.created_at as string | undefined)?.slice(0, 10) ?? ""}
                    </td>
                    <td className="mono sub" title={trainTip(r)}>
                      {r.run_id.slice(0, 12)}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      ))}
    </>
  );
}

/** Hover text for the run id: the training knobs that produced this checkpoint. */
function trainTip(r: Run): string | undefined {
  const prov = r.provenance as Record<string, unknown> | undefined;
  if (!prov) return r.run_id;
  const tc = prov.train_config as Record<string, unknown> | undefined;
  const step = prov.train_step_measured;
  const parts = tc
    ? Object.entries(tc)
        .filter(([, v]) => v != null)
        .map(([k, v]) => `${k}=${v}`)
    : Object.entries(prov)
        .filter(([k]) => k !== "train_step_measured")
        .map(([k, v]) => `${k}=${String(v).slice(-30)}`);
  return `${r.run_id}\n${step != null ? `step ${step} · ` : ""}${parts.join(" · ")}`;
}

export function shortModel(m?: string): string {
  if (!m) return "";
  if (m.startsWith("tinker://")) return m.slice(-24);
  return m.split("/").pop() ?? m;
}
