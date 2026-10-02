"""Read-only HTTP API over a Store.

create_app() builds the FastAPI app and `flipbook serve` runs it
compare results are LRU-cached since they're pure functions of the store.
"""

import json
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flipbook.budget import budget
from flipbook.stats import compare
from flipbook.store import Store

# sample rows carry token_ids/logprobs/messages
_HEAVY_COLS = {"token_ids", "token_logprobs", "messages"}


def _rows(table: pa.Table, drop: set[str] | None = None) -> list[dict]:
    cols = [c for c in table.column_names if not (drop and c in drop)]
    out = table.select(cols).to_pylist()
    # parquet surfaces numpy scalars and NaN
    return json.loads(json.dumps(out, default=str))


def create_app(store_path: str | Path) -> FastAPI:
    store = Store(str(store_path))
    app = FastAPI(title="flipbook", docs_url="/api/docs")

    # cache paired stats, expensiveish
    @lru_cache(maxsize=32)
    def _compare(a: str, b: str) -> dict:
        from dataclasses import asdict
        return asdict(compare(store, a, b))

    @app.get("/api/manifests")
    def manifests() -> list[dict]:
        out = []
        for f in sorted((store.path / "manifests").glob("*.json")):
            doc = json.loads(f.read_text())
            doc["n_rows"] = len(store.manifest_rows(doc["manifest_hash"]))
            out.append(doc)
        return out

    @app.get("/api/runs")
    def runs(study: str | None = None) -> list[dict]:
        return json.loads(json.dumps(store.runs(study), default=str))

    @app.get("/api/runs/{run_id}/samples")
    def samples(run_id: str, full: bool = False) -> list[dict]:
        tbl = store.samples(run_id)
        # store.samples returns an empty table for unknown ids
        # empty 200 would read as "ran, zero rows", so check the file directly
        if tbl.num_rows == 0 and not (store.path / "samples" / f"{run_id}.parquet").exists():
            raise HTTPException(404, f"no run {run_id}")
        return _rows(tbl, drop=None if full else _HEAVY_COLS)

    @app.get("/api/compare")
    def compare_runs(a: str = Query(...), b: str = Query(...)) -> dict:
        for rid in (a, b):
            if not (store.path / "samples" / f"{rid}.parquet").exists():
                raise HTTPException(404, f"no run {rid}")
        return _compare(a, b)

    @app.get("/api/divergence/pairs")
    def divergence_pairs() -> list[dict]:
        d = store.path / "divergence"
        if not d.exists():
            return []
        return [
            {"base": f.stem.split("__")[0], "ckpt": f.stem.split("__")[1]}
            for f in sorted(d.glob("*.parquet"))
        ]

    @app.get("/api/divergence")
    def divergence(base: str = Query(...), ckpt: str = Query(...)) -> list[dict]:
        f = store.path / "divergence" / f"{base}__{ckpt}.parquet"
        if not f.exists():
            raise HTTPException(404, f"no divergence for {base} vs {ckpt}")
        return _rows(store.divergence(base, ckpt))

    @app.get("/api/effort/runs")
    def effort_runs() -> list[dict]:
        d = store.path / "effort"
        if not d.exists():
            return []
        out = []
        for f in sorted(d.glob("*.parquet")):
            run, e = f.stem.split("__")
            lo, hi = e.split("_")
            out.append({"run": run, "e_low": float(lo), "e_high": float(hi)})
        return out

    @app.get("/api/effort")
    def effort(run: str = Query(...), pair: str = Query(...)) -> list[dict]:
        try:
            lo, hi = (float(x) for x in pair.split(","))
        except ValueError:
            raise HTTPException(400, "pair must be e_low,e_high")
        f = store.path / "effort" / f"{run}__{lo}_{hi}.parquet"
        if not f.exists():
            raise HTTPException(404, f"no effort data for {run} @ {pair}")
        return _rows(store.effort(run, lo, hi))

    @app.get("/api/metrics")
    def metrics(study: str = Query(...)) -> list[dict]:
        f = store.path / "training_metrics" / f"{study}.parquet"
        if not f.exists():
            raise HTTPException(404, f"no metrics for study {study}")
        return _rows(store.training_metrics(study))

    @app.get("/api/studies")
    def studies() -> list[str]:
        d = store.path / "training_metrics"
        return sorted(f.stem for f in d.glob("*.parquet")) if d.exists() else []

    @app.get("/api/budget/{run_id}")
    def run_budget(run_id: str) -> dict:
        from dataclasses import asdict
        if not (store.path / "samples" / f"{run_id}.parquet").exists():
            raise HTTPException(404, f"no run {run_id}")
        return asdict(budget(store, run_id))

    # serve the built frontend if web/dist exists (npm run build).
    dist = Path(__file__).parent.parent.parent / "web" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        # registered last on purpose: hash-routed paths like /#compare never
        # hit it, but deep links into the SPA fall through to index.html
        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            f = dist / path
            return FileResponse(f if f.is_file() else dist / "index.html")

    return app


def serve(store_path: str | Path, host: str = "127.0.0.1", port: int = 8484) -> None:
    import uvicorn

    uvicorn.run(create_app(store_path), host=host, port=port)
