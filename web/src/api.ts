export interface Run {
  run_id: string;
  manifest_hash?: string;
  manifest?: string;
  model?: string;
  effort?: number | null;
  k?: number;
  label?: string | null;
  study?: string | null;
  provenance?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface Manifest {
  manifest_hash: string;
  name: string;
  n_rows: number;
  [k: string]: unknown;
}

export interface Flip {
  row_id: string;
  p_a: number;
  p_b: number;
  kind: "regression" | "gain";
  hard: boolean;
}

export interface PairReport {
  run_a: string;
  run_b: string;
  n_pairs: number;
  pairs: string[];
  excluded: { row_id: string; reason: string }[];
  acc_a: number;
  acc_b: number;
  delta: number;
  delta_ci: [number, number];
  agreement: {
    both_right: number;
    both_wrong: number;
    a_only: number;
    b_only: number;
  };
  flips: Flip[];
  tokens: Record<string, { mean: number; p50: number; p90: number; p99: number }>;
  delta_tokens: number;
  delta_tokens_ci: [number, number];
  truncation_rate_a: number;
  truncation_rate_b: number;
  cost_a: number;
  cost_b: number;
  failures: Record<string, Record<string, number>>;
}

export interface DivergenceRow {
  row_id: string;
  sample_idx: number;
  delta: number[];
  sum_nats: number;
  mean_nats: number;
  divergence_pos: number | null;
  win_argmin: number | null;
  p_skip_base: number;
  p_skip_ckpt: number;
  cost_usd: number;
}

export interface MetricRow {
  study: string;
  step: number;
  key: string;
  value: number;
}

export interface SampleRow {
  row_id: string;
  sample_idx: number;
  text: string;
  gen_tokens: number;
  stop_reason: string;
  verdict: number | null;
  failure_kind: string | null;
  est_cost_usd: number | null;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  manifests: () => get<Manifest[]>("/api/manifests"),
  runs: () => get<Run[]>("/api/runs"),
  samples: (run: string) => get<SampleRow[]>(`/api/runs/${run}/samples`),
  compare: (a: string, b: string) => get<PairReport>(`/api/compare?a=${a}&b=${b}`),
  divergence: (base: string, ckpt: string) =>
    get<DivergenceRow[]>(`/api/divergence?base=${base}&ckpt=${ckpt}`),
  divergencePairs: () => get<{ base: string; ckpt: string }[]>("/api/divergence/pairs"),
  studies: () => get<string[]>("/api/studies"),
  metrics: (study: string) => get<MetricRow[]>(`/api/metrics?study=${study}`),
};
