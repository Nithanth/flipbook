"""On-disk store: Parquet files deduped with idempotency + DuckDB engine for reads

"""

from __future__ import annotations

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
            return
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
            del r["manifest_hash"]
        return rows

    def put_run(self, run: dict) -> None:
        _write_json(run, self._dir("runs") / f"{run['run_id']}.json")

    def runs(self, study: str | None = None) -> list[dict]:
        runs = [
            json.loads(f.read_text()) for f in sorted((self.path / "runs").glob("*.json"))
        ]
        return [r for r in runs if study is None or r.get("study") == study]

    def put_samples(self, run_id: str, rows: list[dict]) -> None:
        tbl = pa.Table.from_pylist(rows, schema=SAMPLES)
        f = self._dir("samples") / f"{run_id}.parquet"
        if f.exists():
            tbl = _dedupe(pa.concat_tables([pq.read_table(f), tbl]))
        _write_parquet(tbl, f)

    def samples(self, run_id: str) -> pa.Table:
        f = self.path / "samples" / f"{run_id}.parquet"
        return pq.read_table(f) if f.exists() else SAMPLES.empty_table()

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


def _write_parquet(tbl: pa.Table, path: Path) -> None:
    tmp = path.parent / (path.name + ".tmp")
    pq.write_table(tbl, tmp)
    os.replace(tmp, path)


def _write_json(doc: dict, path: Path) -> None:
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


def _dedupe(tbl: pa.Table) -> pa.Table:
    seen, keep = set(), []
    for i, (rid, s) in enumerate(zip(tbl["row_id"].to_pylist(), tbl["sample_idx"].to_pylist())):
        if (rid, s) not in seen:
            seen.add((rid, s))
            keep.append(i)
    return tbl.take(pa.array(keep))
