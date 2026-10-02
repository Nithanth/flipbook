import { useEffect, useMemo, useState } from "react";
import { api, type MetricRow } from "../api";

export default function Study() {
  const [studies, setStudies] = useState<string[]>([]);
  const [study, setStudy] = useState("");
  const [rows, setRows] = useState<MetricRow[]>([]);

  useEffect(() => {
    api.studies().then((ss) => {
      setStudies(ss);
      if (ss.length) setStudy(ss[0]);
    });
  }, []);

  useEffect(() => {
    if (study) api.metrics(study).then(setRows);
  }, [study]);

  const keys = useMemo(() => [...new Set(rows.map((r) => r.key))].sort(), [rows]);
  const [key, setKey] = useState<string>("");
  useEffect(() => {
    if (keys.length && !keys.includes(key)) setKey(keys[0]);
  }, [keys, key]);

  const series = rows.filter((r) => r.key === key);

  return (
    <>
      <h1>study</h1>
      <div className="row">
        <label>
          study{" "}
          <select value={study} onChange={(e) => setStudy(e.target.value)}>
            {studies.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label>
          metric{" "}
          <select value={key} onChange={(e) => setKey(e.target.value)}>
            {keys.map((k) => (
              <option key={k}>{k}</option>
            ))}
          </select>
        </label>
      </div>
      {series.length > 1 ? <Chart data={series} /> : <p>no data</p>}
    </>
  );
}

function Chart({ data }: { data: MetricRow[] }) {
  const w = 720;
  const h = 260;
  const pad = 36;
  const xs = data.map((d) => d.step);
  const ys = data.map((d) => d.value);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x: number) => pad + ((x - x0) / Math.max(1, x1 - x0)) * (w - 2 * pad);
  const sy = (y: number) => h - pad - ((y - y0) / Math.max(1e-12, y1 - y0)) * (h - 2 * pad);
  const path = data.map((d, i) => `${i ? "L" : "M"}${sx(d.step)},${sy(d.value)}`).join(" ");
  return (
    <svg width={w} height={h} className="chart">
      <line x1={pad} x2={w - pad} y1={h - pad} y2={h - pad} stroke="#444" />
      <line x1={pad} x2={pad} y1={pad} y2={h - pad} stroke="#444" />
      <path d={path} fill="none" stroke="#4b8be5" strokeWidth={1.5} />
      {data.map((d) => (
        <circle key={d.step} cx={sx(d.step)} cy={sy(d.value)} r={2.5} fill="#4b8be5">
          <title>{`step ${d.step}: ${d.value}`}</title>
        </circle>
      ))}
      <text x={pad} y={h - 8} fill="#888" fontSize={11}>{x0}</text>
      <text x={w - pad - 20} y={h - 8} fill="#888" fontSize={11}>{x1}</text>
      <text x={6} y={pad} fill="#888" fontSize={11}>{y1.toPrecision(3)}</text>
      <text x={6} y={h - pad} fill="#888" fontSize={11}>{y0.toPrecision(3)}</text>
    </svg>
  );
}
