"""Read-only HTTP API over a Store.

create_app() builds the FastAPI app and `flipbook serve` runs it
compare results are LRU-cached since they're pure functions of the store.
"""

import json
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flipbook.budget import budget
from flipbook.decode import thinking as _thinking
from flipbook.decode import tokenizer as _tokenizer
from flipbook.graders import strip_control_tokens
from flipbook.stats import compare
from flipbook.store import Store

# sample rows carry token_ids/logprobs/messages
_HEAVY_COLS = {"token_ids", "token_logprobs", "messages"}


def _rows(table: pa.Table, drop: set[str] | None = None) -> list[dict]:
    cols = [c for c in table.column_names if not (drop and c in drop)]
    out = table.select(cols).to_pylist()
    # parquet surfaces numpy scalars and NaN
    return json.loads(json.dumps(out, default=str))


def _norm_run(r: dict) -> dict:
    # evaluator stamps provenance.train_step_measured; lift it so consumers
    # can sort/group on a single field
    if r.get("train_step") is None:
        r["train_step"] = (r.get("provenance") or {}).get("train_step_measured")
    return r


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
        out = []
        for r in store.runs(study):
            r = _norm_run(r)
            t = store.samples(r["run_id"])
            vs = [v for v in t.column("verdict").to_pylist() if v is not None]
            r["acc"] = sum(vs) / len(vs) if vs else None
            out.append(r)
        return json.loads(json.dumps(out, default=str))

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
        report = _compare(a, b)
        if blocks := report["comparability"].get("blocks"):
            raise HTTPException(422, "; ".join(blocks))
        return report

    @app.get("/api/compare/row")
    def compare_row(
        a: str = Query(...), b: str = Query(...), row: str = Query(...)
    ) -> dict:
        for rid in (a, b):
            if not (store.path / "samples" / f"{rid}.parquet").exists():
                raise HTTPException(404, f"no run {rid}")
        run_a = next((r for r in store.runs() if r["run_id"] == a), None) or {}
        mh = run_a.get("manifest_hash")
        mrows = (
            store.manifest_rows(mh)
            if mh and (store.path / "manifest_rows" / f"{mh}.parquet").exists()
            else []
        )
        mrow = next((r for r in mrows if r["row_id"] == row), None)
        if not mrow:
            raise HTTPException(404, f"no row {row} in manifest for run {a}")
        keep = {"sample_idx", "text", "gen_tokens", "stop_reason",
                "verdict", "extracted", "failure_kind", "grade_note"}

        def side(rid: str) -> list[dict]:
            run = next((r for r in store.runs() if r["run_id"] == rid), None) or {}
            model_id = run.get("model_id") or run.get("model")
            tbl = store.samples(rid)
            # token_ids is the heavy column; filter to this row before materializing
            tbl = tbl.filter(pc.equal(tbl.column("row_id"), row))
            out = []
            for r in _rows(tbl):
                s = {k: r.get(k) for k in keep}
                s["text_clean"] = strip_control_tokens(r.get("text") or "").strip()
                s["thinking"] = _thinking(model_id, r.get("token_ids"), r.get("text"))
                out.append(s)
            return sorted(out, key=lambda r: r["sample_idx"])

        run_b = next((r for r in store.runs() if r["run_id"] == b), None) or {}
        return {
            "row_id": row,
            "question": mrow["messages"],
            "answer": mrow["gold"],
            "k_a": run_a.get("k"),
            "k_b": run_b.get("k"),
            "a": side(a),
            "b": side(b),
        }

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

    @app.get("/api/divergence/trace")
    def divergence_trace(
        base: str = Query(...),
        ckpt: str = Query(...),
        row: str = Query(...),
        sample: int = Query(...),
    ) -> dict:
        # pairs each delta with the baseline's token string so the UI can render
        # the trace as colored text — 404s (not a sparse delta array) when the
        # store predates token_ids
        if not (store.path / "divergence" / f"{base}__{ckpt}.parquet").exists():
            raise HTTPException(404, f"no divergence for {base} vs {ckpt}")
        drow = next(
            (
                r
                for r in _rows(store.divergence(base, ckpt))
                if r["row_id"] == row and r["sample_idx"] == sample
            ),
            None,
        )
        if not drow:
            raise HTTPException(404, f"no divergence row {row}:{sample}")
        srow = next(
            (
                r
                for r in _rows(store.samples(base))
                if r["row_id"] == row and r["sample_idx"] == sample
            ),
            None,
        )
        ids = (srow or {}).get("token_ids")
        if not ids:
            raise HTTPException(404, "no baseline token ids for this sample")
        run = next((r for r in store.runs() if r["run_id"] == base), {})
        model_id = run.get("model_id") or run.get("model")
        if not model_id:
            raise HTTPException(404, f"no model recorded on run {base}")
        tok = _tokenizer(model_id)
        return {
            "tokens": [
                {"t": tok.decode([int(i)]), "d": d}
                for i, d in zip(ids, drow["delta"])
            ]
        }

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
        names = {f.stem for f in d.glob("*.parquet")} if d.exists() else set()
        names |= {r["study"] for r in store.runs() if r.get("study")}
        return sorted(names)

    @app.get("/api/studies/{name}")
    def study_detail(name: str) -> dict:
        sruns = sorted(
            (_norm_run(r) for r in store.runs(name)),
            key=lambda r: r.get("train_step") if r.get("train_step") is not None else -1,
        )
        if not sruns:
            raise HTTPException(404, f"no study {name}")
        metrics: dict[str, list[dict]] = {}
        if (store.path / "training_metrics" / f"{name}.parquet").exists():
            for r in _rows(store.training_metrics(name)):
                metrics.setdefault(r["key"], []).append({"step": r["step"], "value": r["value"]})
        effort = {}
        d = store.path / "effort"
        if d.exists():
            for f in sorted(d.glob("*.parquet")):
                run, e = f.stem.split("__")
                lo, hi = (float(x) for x in e.split("_"))
                t = store.effort(run, lo, hi)
                if t.num_rows:
                    gaps = [g for g in t.column("gap_nats").to_pylist() if g is not None]
                    if gaps:
                        effort[run] = sum(gaps) / len(gaps)
        return {"study": name, "runs": json.loads(json.dumps(sruns, default=str)),
                "metrics": metrics, "effort_gap": effort}

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
