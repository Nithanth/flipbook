export interface Run {
  run_id: string;
  manifest_hash?: string;
  manifest?: string;
  model?: string;
  effort?: number | null;
  k?: number;
  label?: string | null;
  study?: string | null;
  train_step?: number | null;
  provenance?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface StudyDetail {
  study: string;
  runs: Run[];
  metrics: Record<string, { step: number; value: number }[]>;
  effort_gap: Record<string, number>;
}

export interface Manifest {
  manifest_hash: string;
  name: string;
  n_rows: number;
  [k: string]: unknown;
}

export interface DivergencePair {
  base: string;
  ckpt: string;
  self: boolean;
  median_nats: number;
  max_abs_nats: number;
  noise_floor: number | null;
}

export interface Flip {
  row_id: string;
  p_a: number;
  p_b: number;
  kind: "regression" | "gain";
  hard: boolean;
  // truncated question text (newer servers); null on old stores
  q?: string | null;
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
  comparability?: { ok: boolean; blocks: string[]; warnings: string[]; token_views: boolean };
  // absent when an older server serves the pair report
  cells?: {
    row_id: string;
    p_a: number | null;
    p_b: number | null;
    cell: "both_right" | "both_wrong" | "a_only" | "b_only" | "excluded";
    q?: string | null;
  }[];
  failure_rows_b?: Record<string, string[]>;
  passn?: Record<
    string,
    { a: number; b: number; delta: number; delta_ci: [number, number] }
  >;
}

export interface DivergenceRow {
  row_id: string;
  sample_idx: number;
  n?: number | null;
  delta: number[];
  sum_nats: number;
  mean_nats: number;
  divergence_pos: number | null;
  win_argmin: number | null;
  p_skip_base: number;
  p_skip_ckpt: number;
  cost_usd: number;
  // truncated question text (newer servers)
  q?: string | null;
}

export interface BranchResult {
  pos: number;
  base_tok: string;
  base_tok_lp: number;
  ckpt_lp_on_base_tok: number;
  ckpt_first_tok: string;
  ckpt_first_tok_lp: number | null;
  base_cont: string;
  ckpt_cont: string;
}

export interface MetricRow {
  study: string;
  step: number;
  key: string;
  value: number;
}

export interface RowSample {
  sample_idx: number;
  text: string;
  gen_tokens: number;
  stop_reason: string;
  verdict: number | null;
  extracted: string | null;
  failure_kind: string | null;
  // what the grader reported - the "why" for custom graders and judges
  grade_note?: string | null;
  // decoded generation before the final message; null when token_ids are absent
  thinking?: string | null;
  // text with renderer control tokens stripped
  text_clean?: string;
}

export interface RowDetail {
  row_id: string;
  question: { role: string; content: string }[];
  answer: string;
  k_a?: number | null;
  k_b?: number | null;
  a: RowSample[];
  b: RowSample[];
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
  if (!r.ok) {
    // FastAPI errors are {"detail": "..."}; show the message, not the envelope
    const body = await r.text();
    let detail = body;
    try {
      detail = JSON.parse(body).detail ?? body;
    } catch {
      /* not JSON */
    }
    throw new Error(`${r.status} ${detail}`);
  }
  // a non-JSON 200 means the SPA fallback answered - the running server
  // predates this endpoint
  if (!r.headers.get("content-type")?.includes("application/json")) {
    throw new Error(
      `${path}: the server returned HTML, not JSON - it's probably running an older build; restart \`flipbook serve\``,
    );
  }
  return r.json();
}

export const api = {
  manifests: () => get<Manifest[]>("/api/manifests"),
  runs: () => get<Run[]>("/api/runs"),
  samples: (run: string) => get<SampleRow[]>(`/api/runs/${run}/samples`),
  compare: (a: string, b: string) => get<PairReport>(`/api/compare?a=${a}&b=${b}`),
  compareRow: (a: string, b: string, row: string) =>
    get<RowDetail>(
      `/api/compare/row?a=${a}&b=${b}&row=${encodeURIComponent(row)}`,
    ),
  divergence: (base: string, ckpt: string) =>
    get<DivergenceRow[]>(`/api/divergence?base=${base}&ckpt=${ckpt}`),
  divergencePairs: () => get<DivergencePair[]>("/api/divergence/pairs"),
  divergenceTrace: (base: string, ckpt: string, row: string, sample: number) =>
    get<{ tokens: { t: string; d: number }[] }>(
      `/api/divergence/trace?base=${base}&ckpt=${ckpt}&row=${encodeURIComponent(row)}&sample=${sample}`,
    ),
  divergenceBranch: (base: string, ckpt: string, row: string, sample: number, pos: number) =>
    get<BranchResult>(
      `/api/divergence/branch?base=${base}&ckpt=${ckpt}&row=${encodeURIComponent(row)}&sample=${sample}&pos=${pos}`,
    ),
  studies: () => get<string[]>("/api/studies"),
  study: (name: string) => get<StudyDetail>(`/api/studies/${name}`),
  metrics: (study: string) => get<MetricRow[]>(`/api/metrics?study=${study}`),
};
