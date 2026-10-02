import { useEffect, useState } from "react";
import { api, type Run } from "../api";

export default function Runs() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="err">{err}</p>;
  if (!runs) return <p>loading…</p>;

  return (
    <>
      <h1>runs</h1>
      <table>
        <thead>
          <tr>
            <th>run</th>
            <th>model</th>
            <th>label</th>
            <th>effort</th>
            <th>k</th>
            <th>manifest</th>
            <th>study</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id}>
              <td className="mono">{r.run_id.slice(0, 12)}</td>
              <td className="mono" title={r.model}>{shortModel(r.model)}</td>
              <td>{r.label ?? ""}</td>
              <td>{r.effort ?? ""}</td>
              <td>{r.k ?? ""}</td>
              <td className="mono">{(r.manifest_hash ?? "").slice(0, 10)}</td>
              <td>{r.study ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function shortModel(m?: string): string {
  if (!m) return "";
  if (m.startsWith("tinker://")) return m.slice(-24);
  return m.split("/").pop() ?? m;
}
