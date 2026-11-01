"""On-disk store: Parquet files deduped with idempotency + DuckDB engine for reads

Writes are tmp+rename atomic, a reader sees old or new, never torn.
"""


import json
import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_ROWS = pa.schema(
    [
        ("manifest_hash", pa.string()),
        ("row_id", pa.string()),
        ("benchmark", pa.string()),
        ("messages", pa.string()),  # JSON
        ("gold", pa.string()),
        ("grader_id", pa.string()),
        ("source_ids", pa.string()),  # JSON
    ]
)

SAMPLES = pa.schema(
    [
        ("run_id", pa.string()),
        ("row_id", pa.string()),
        ("sample_idx", pa.int32()),
        ("text", pa.string()),
        ("prompt_tokens", pa.int32()),
        ("gen_tokens", pa.int32()),
        ("stop_reason", pa.string()),
        ("verdict", pa.float64()),  # null on error
        ("extracted", pa.string()),
        ("failure_kind", pa.string()),  # null|truncation|parse|wrong_answer|error
        ("grade_note", pa.string()),
        ("error", pa.string()),
        ("est_cost_usd", pa.float64()),
        ("token_ids", pa.list_(pa.int32())),
        ("token_logprobs", pa.list_(pa.float32())),
    ]
)

DIVERGENCE = pa.schema(
    [
        ("base_run_id", pa.string()),
        ("ckpt_run_id", pa.string()),
        ("row_id", pa.string()),
        ("sample_idx", pa.int32()),
        ("prompt_len", pa.int32()),
        ("n", pa.int32()),
        ("lp_base", pa.list_(pa.float32())),
        ("lp_ckpt", pa.list_(pa.float32())),
        ("delta", pa.list_(pa.float32())),
        ("sum_nats", pa.float64()),
        ("mean_nats", pa.float64()),
        ("divergence_pos", pa.int32()),  # null if the -tau cumsum never crossed
        ("win_argmin", pa.int32()),  # argmin of windowed mean
        ("p_skip_base", pa.float64()),
        ("p_skip_ckpt", pa.float64()),
        ("cost_usd", pa.float64()),
    ]
)

EFFORT = pa.schema(
    [
        ("run_id", pa.string()),
        ("row_id", pa.string()),
        ("sample_idx", pa.int32()),
        ("e_low", pa.float64()),
        ("e_high", pa.float64()),
        ("lp_low_sum", pa.float64()),
        ("lp_high_sum", pa.float64()),
        ("gap_nats", pa.float64()),
        ("n", pa.int32()),
        ("cost_usd", pa.float64()),
    ]
)

TRAINING_METRICS = pa.schema(
    [
        ("study", pa.string()),
        ("step", pa.int64()),
        ("key", pa.string()),
        ("value", pa.float64()),
    ]
)


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _dir(self, name: str) -> Path:
        d = self.path / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put_manifest(self, doc: dict, rows: list[dict]) -> None:
        h = doc["manifest_hash"]
        if (self.path / "manifests" / f"{h}.json").exists():
            return  # same hash = same manifest
        _write_json(doc, self._dir("manifests") / f"{h}.json")
        tbl = pa.Table.from_pylist(
            [
                {
                    **r,
                    "manifest_hash": h,
                    "messages": json.dumps(r["messages"]),
                    "source_ids": json.dumps(r["source_ids"]),
                }
                for r in rows
            ],
            schema=MANIFEST_ROWS,
        )
        _write_parquet(tbl, self._dir("manifest_rows") / f"{h}.parquet")

    def manifest_doc(self, name_or_hash: str) -> dict | None:
        for f in sorted((self.path / "manifests").glob("*.json")):
            doc = json.loads(f.read_text())
            if (
                doc["name"] == name_or_hash
                or doc["manifest_hash"] == name_or_hash
                or doc["manifest_hash"].startswith(name_or_hash)
            ):
                return doc
        return None

    def manifest_rows(self, manifest_hash: str) -> list[dict]:
        f = self.path / "manifest_rows" / f"{manifest_hash}.parquet"
        rows = pq.read_table(f).to_pylist()
        for r in rows:
            r["messages"] = json.loads(r["messages"])
            r["source_ids"] = json.loads(r["source_ids"])
            del r["manifest_hash"]  # keep rows row-shaped for re-freezing
        return rows

    def put_run(self, run: dict) -> None:
        _write_json(run, self._dir("runs") / f"{run['run_id']}.json")

    def runs(self, study: str | None = None) -> list[dict]:
        runs = [
            json.loads(f.read_text()) for f in sorted((self.path / "runs").glob("*.json"))
        ]
        return [r for r in runs if study is None or r.get("study") == study]

    def put_samples(self, run_id: str, rows: list[dict]) -> None:
        _merge_write(self._dir("samples") / f"{run_id}.parquet", SAMPLES, rows, ("row_id", "sample_idx"))

    def samples(self, run_id: str) -> pa.Table:
        f = self.path / "samples" / f"{run_id}.parquet"
        return pq.read_table(f) if f.exists() else SAMPLES.empty_table()

    def put_divergence(self, base_run_id: str, ckpt_run_id: str, rows: list[dict]) -> None:
        f = self._dir("divergence") / f"{base_run_id}__{ckpt_run_id}.parquet"
        _merge_write(f, DIVERGENCE, rows, ("row_id", "sample_idx"))

    def divergence(self, base_run_id: str, ckpt_run_id: str) -> pa.Table:
        f = self.path / "divergence" / f"{base_run_id}__{ckpt_run_id}.parquet"
        return pq.read_table(f) if f.exists() else DIVERGENCE.empty_table()

    def put_effort(self, run_id: str, e_low: float, e_high: float, rows: list[dict]) -> None:
        f = self._dir("effort") / f"{run_id}__{e_low}_{e_high}.parquet"
        _merge_write(f, EFFORT, rows, ("row_id", "sample_idx"))

    def effort(self, run_id: str, e_low: float, e_high: float) -> pa.Table:
        f = self.path / "effort" / f"{run_id}__{e_low}_{e_high}.parquet"
        return pq.read_table(f) if f.exists() else EFFORT.empty_table()

    def put_training_metrics(self, study: str, records: list[dict]) -> None:
        _merge_write(
            self._dir("training_metrics") / f"{study}.parquet",
            TRAINING_METRICS, records, ("step", "key"),
        )

    def training_metrics(self, study: str) -> pa.Table:
        f = self.path / "training_metrics" / f"{study}.parquet"
        return pq.read_table(f) if f.exists() else TRAINING_METRICS.empty_table()

    def has(self, run_id: str, row_id: str, sample_idx: int) -> bool:
        f = self.path / "samples" / f"{run_id}.parquet"
        if not f.exists():
            return False
        t = self.query(
            "select 1 from read_parquet($f) where row_id = $rid and sample_idx = $s limit 1",
            f=str(f), rid=row_id, s=sample_idx,
        )
        return t.num_rows > 0

    def query(self, sql: str, **params) -> pa.Table:
        return duckdb.execute(sql, params).to_arrow_table()


def _merge_write(f: Path, schema: pa.Schema, rows: list[dict], key: tuple[str, ...]) -> None:
    tbl = pa.Table.from_pylist(rows, schema=schema)
    if f.exists():
        tbl = _dedupe(pa.concat_tables([pq.read_table(f), tbl], promote_options="default"), key)
    _write_parquet(tbl, f)


def _write_parquet(tbl: pa.Table, path: Path) -> None:
    tmp = path.parent / (path.name + ".tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, path)


def _write_json(doc: dict, path: Path) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


def _dedupe(tbl: pa.Table, key: tuple[str, ...]) -> pa.Table:
    # stored rows are never overwritten by a rewrite
    cols = [tbl[k].to_pylist() for k in key]
    seen, keep = set(), []
    for i, ks in enumerate(zip(*cols)):
        if ks not in seen:
            seen.add(ks)
            keep.append(i)
    return tbl.take(pa.array(keep))
