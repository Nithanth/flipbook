import { useEffect, useState } from "react";
import { api, type Run } from "../api";
import { runLabel } from "../runLabel";

export default function Runs() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="err">{err}</p>;
  if (!runs) return <p>loading…</p>;

  const studies = [...new Set(runs.map((r) => r.study ?? "ungrouped"))].sort(
    (a, b) =>
      runs.filter((r) => (r.study ?? "ungrouped") === b).length -
      runs.filter((r) => (r.study ?? "ungrouped") === a).length,
  );

  return (
    <>
      <h1>runs</h1>
      <p className="page-sub">every evaluation in the store, grouped by study</p>
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
                <th>label</th>
                <th>run</th>
                <th>model</th>
                <th>effort</th>
                <th>k</th>
                <th>manifest</th>
              </tr>
            </thead>
            <tbody>
              {runs
                .filter((r) => (r.study ?? "ungrouped") === s)
                .sort((x, y) => (x.train_step ?? -1) - (y.train_step ?? -1))
                .map((r) => (
                  <tr key={r.run_id}>
                    <td>{runLabel(r)}</td>
                    <td className="mono">{r.run_id.slice(0, 12)}</td>
                    <td className="mono" title={r.model}>{shortModel(r.model)}</td>
                    <td>{r.effort ?? ""}</td>
                    <td>{r.k ?? ""}</td>
                    <td className="mono">{(r.manifest_hash ?? "").slice(0, 10)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      ))}
    </>
  );
}

export function shortModel(m?: string): string {
  if (!m) return "";
  if (m.startsWith("tinker://")) return m.slice(-24);
  return m.split("/").pop() ?? m;
}
