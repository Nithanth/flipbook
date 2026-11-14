# Store

The store is a directory, not a service. Parquet files are the truth; DuckDB
is an in-process query engine over `read_parquet('store/samples/*.parquet')`
globs; never a server to run.

Layout:

``` text
flipbook_store/
  manifests/       one JSON per manifest: hash, name, n_rows, spec
  manifest_rows/   one Parquet per manifest: the rows
  runs/            one JSON per run: config, provenance, versions
  samples/         one Parquet per run: one row per (row_id, sample_idx)
  rollouts/        one Parquet per study: one row per rollout step
  rollout_texts/   one Parquet per study: sampled transcripts, keyed by log position
```

## Why Parquet + DuckDB

Two properties made this the choice over SQLite or a real database:

- **Zero ops.** The store is a directory of files; it diffs, rsyncs, and ships
  to S3. There is nothing to start, and no schema migrations — a reader can
  always open files written by any earlier version.
- **Columnar lists.** Sample rows carry per-token arrays (`token_ids`,
  `token_logprobs`). Parquet stores them natively; DuckDB can aggregate over
  them without decoding rows. The same schema drops into ClickHouse
  (MergeTree ordered by `(manifest_hash, row_id)`) unchanged if the store
  ever outgrows a laptop.

## Invariants

- **Atomic writes.** Every file goes `tmp` + `os.replace`. A concurrent
  reader sees the old file or the new one, never a partial write.
- **Idempotency is a file-existence question.** `put_samples` merges and
  dedupes on `(row_id, sample_idx)`; `has()` is the skip check.
- **Identity is hashed, not labeled.** `manifest_hash` and `run_id` are
  sha256 fingerprints of content; names and labels are display metadata and
  never participate in keys.
- **Runs have a lifecycle.** `eval` writes the run record with
  `state: "running"` before the first paid call, flushes samples every 25
  rows, and finishes with `state: "done"` plus measured results in
  `provenance`. A killed eval keeps its completed cells and a rerun resumes
  via dedupe; readers treat `state == "running"` as partial progress, not a
  finished point.

DuckDB opens a fresh in-memory connection per query, so readers hold no
shared state and need no locks.

## Training rollouts

Import cookbook logs with `flipbook import-rollouts <log dir> --study <name>`.
The Study page shows reward, final-step metrics, response lengths, and a
transcript browser. Studies with only imported rollouts also appear in the picker.

Rollout summaries have iteration, group, trajectory, and step IDs. Charts
average episodes: token counts sum the steps, total reward is counted once,
and metrics come from the final step. The values table includes the number
of episodes reporting each metric; missing metrics are not counted as zero.

Logtree transcripts have positions rather than matching rollout IDs.
They stay separate from the summaries, with coverage shown for each split
and iteration. Transcript filters use only fields logged with the text;
length sorting uses characters, since these records lack token counts.
Side-by-side iteration lists are not paired comparisons of the same questions.

Chart annotations use trailing three-iteration means: the first crossing
of 0.98 from below, or a drop of 0.10 from the prior peak. They apply only
to metrics with means in [0, 1] at every recorded iteration. These are
inspection hints, not significance tests or evidence of a cause.
