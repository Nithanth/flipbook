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

DuckDB opens a fresh in-memory connection per query, so readers hold no
shared state and need no locks.
