import type { Run } from "./api";

export default function SyntheticNotice({ runs }: { runs: Run[] }) {
  if (!runs.some((r) => r.source === "synthetic")) return null;
  return <p className="card">synthetic demo data — numbers and costs are illustrative</p>;
}
