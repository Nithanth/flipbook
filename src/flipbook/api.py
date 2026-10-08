"""Read-only HTTP API over a Store.

create_app() builds the FastAPI app and `flipbook serve` runs it
compare results are LRU-cached since they're pure functions of the store.
"""

import json
import time
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


def _downsample(xs: list[float], n: int = 512) -> list[float]:
    """Mean-pool-adjacent: keep each bin's largest-magnitude value (with sign),
    so spikes survive the compression that sparklines apply anyway."""
    if len(xs) <= n:
        return xs
    out = []
    step = len(xs) / n
    for i in range(n):
        seg = xs[int(i * step) : int((i + 1) * step)]
        out.append(max(seg, key=abs) if seg else 0.0)
    return out


def create_app(store_path: str | Path) -> FastAPI:
    store = Store(str(store_path))
    app = FastAPI(title="flipbook", docs_url="/api/docs")

    # short-TTL read cache: repeated row clicks shouldn't re-read parquets, but
    # an eval landing mid-session should still show up without a restart
    _cache: dict[str, tuple[float, object]] = {}

    def _cached(key: str, fn, ttl: float = 5.0):
        ent = _cache.get(key)
        if ent and time.time() - ent[0] < ttl:
            return ent[1]
        v = fn()
        _cache[key] = (time.time(), v)
        return v

    def _samples(rid: str):
        return _cached(f"samples:{rid}", lambda: store.samples(rid))

    def _mrows(mh: str):
        return _cached(f"mrows:{mh}", lambda: store.manifest_rows(mh))

    def _run_list(study: str | None = None):
        all_runs = _cached("runs", store.runs)
        return [r for r in all_runs if not study or r.get("study") == study]

    def _divergence(a: str, b: str):
        return _cached(f"div:{a}__{b}", lambda: store.divergence(a, b))

    # cache paired stats, expensiveish - keyed on sample counts so an eval
    # still accumulating mid-session re-reads instead of serving stale rows
    @lru_cache(maxsize=32)
    def _compare(a: str, b: str, sig: tuple[int, int]) -> dict:
        from dataclasses import asdict
        return asdict(compare(store, a, b))

    def _compare_live(a: str, b: str) -> dict:
        return _compare(a, b, (_samples(a).num_rows, _samples(b).num_rows))

    @app.get("/api/manifests")
    def manifests() -> list[dict]:
        out = []
        for f in sorted((store.path / "manifests").glob("*.json")):
            doc = json.loads(f.read_text())
            doc["n_rows"] = len(_mrows(doc["manifest_hash"]))
            out.append(doc)
        return out

    @app.get("/api/runs")
    def runs(study: str | None = None) -> list[dict]:
        out = []
        for r in _run_list(study):
            r = _norm_run(r)
            t = _samples(r["run_id"])
            vs = [v for v in t.column("verdict").to_pylist() if v is not None]
            r["acc"] = sum(vs) / len(vs) if vs else None
            out.append(r)
        return json.loads(json.dumps(out, default=str))

    @app.get("/api/runs/{run_id}/samples")
    def samples(run_id: str, full: bool = False) -> list[dict]:
        tbl = _samples(run_id)
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
        report = _compare_live(a, b)
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
        run_a = next((r for r in _run_list() if r["run_id"] == a), None) or {}
        mh = run_a.get("manifest_hash")
        mrows = (
            _mrows(mh)
            if mh and (store.path / "manifest_rows" / f"{mh}.parquet").exists()
            else []
        )
        mrow = next((r for r in mrows if r["row_id"] == row), None)
        if not mrow:
            raise HTTPException(404, f"no row {row} in manifest for run {a}")
        keep = {"sample_idx", "text", "gen_tokens", "stop_reason",
                "verdict", "extracted", "failure_kind", "grade_note"}

        def side(rid: str) -> list[dict]:
            run = next((r for r in _run_list() if r["run_id"] == rid), None) or {}
            model_id = run.get("model_id") or run.get("model")
            tbl = _samples(rid)
            # token_ids is the heavy column; filter to this row before materializing
            tbl = tbl.filter(pc.equal(tbl.column("row_id"), row))
            out = []
            for r in _rows(tbl):
                s = {k: r.get(k) for k in keep}
                s["text_clean"] = strip_control_tokens(r.get("text") or "").strip()
                s["thinking"] = _thinking(model_id, r.get("token_ids"), r.get("text"))
                out.append(s)
            return sorted(out, key=lambda r: r["sample_idx"])

        run_b = next((r for r in _run_list() if r["run_id"] == b), None) or {}
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
        out = []
        for f in sorted(d.glob("*.parquet")):
            base, ckpt = f.stem.split("__")
            sums = _divergence(base, ckpt).column("sum_nats").to_pylist()
            out.append({
                "base": base,
                "ckpt": ckpt,
                "self": base == ckpt,
                "median_nats": sorted(sums)[len(sums) // 2] if sums else 0.0,
                "max_abs_nats": max((abs(x) for x in sums), default=0.0),
            })
        floor = max((p["max_abs_nats"] for p in out if p["self"]), default=None)
        for p in out:
            p["noise_floor"] = floor
        # most-diverged real pairs first; the A/A noise floor sorts last
        out.sort(key=lambda p: (p["self"], -abs(p["median_nats"])))
        return out

    @app.get("/api/divergence")
    def divergence(base: str = Query(...), ckpt: str = Query(...)) -> list[dict]:
        f = store.path / "divergence" / f"{base}__{ckpt}.parquet"
        if not f.exists():
            raise HTTPException(404, f"no divergence for {base} vs {ckpt}")
        from flipbook.manifest import question_text, truncate_q

        run = next((r for r in _run_list() if r["run_id"] == base), {})
        qtext = {
            r["row_id"]: truncate_q(question_text(r["messages"]))
            for r in _mrows(run.get("manifest_hash") or "")
        }
        # lp arrays are per-row drill-down data (trace/branch fetch them on
        # expand); the table only needs a compressed delta for the sparkline,
        # so don't ship ~2MB of floats per row up front
        out = []
        for r in _rows(_divergence(base, ckpt), drop={"lp_base", "lp_ckpt"}):
            r["delta"] = _downsample(r.get("delta") or [])
            r["q"] = qtext.get(r["row_id"])
            out.append(r)
        return out

    @app.get("/api/divergence/branch")
    async def divergence_branch(
        base: str = Query(...),
        ckpt: str = Query(...),
        row: str = Query(...),
        sample: int = Query(...),
        pos: int = Query(...),
        n: int = Query(96),
    ) -> dict:
        # greedy ckpt continuation from a cut on the base's own trace — the one
        # paid endpoint; ~n sampled tokens per call
        from flipbook.diverge import branch

        try:
            return await branch(store, base, ckpt, row, sample, pos, n)
        except LookupError as e:
            raise HTTPException(404, str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"branch sampling failed: {type(e).__name__}: {e}") from e

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
                for r in _rows(_divergence(base, ckpt))
                if r["row_id"] == row and r["sample_idx"] == sample
            ),
            None,
        )
        if not drow:
            raise HTTPException(404, f"no divergence row {row}:{sample}")
        srow = next(
            (
                r
                for r in _rows(_samples(base))
                if r["row_id"] == row and r["sample_idx"] == sample
            ),
            None,
        )
        ids = (srow or {}).get("token_ids")
        if not ids:
            raise HTTPException(404, "no baseline token ids for this sample")
        run = next((r for r in _run_list() if r["run_id"] == base), {})
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
        names |= {r["study"] for r in _run_list() if r.get("study")}
        return sorted(names)

    @app.get("/api/studies/{name}")
    def study_detail(name: str) -> dict:
        sruns = sorted(
            (_norm_run(r) for r in _run_list(name)),
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


def load_env_file(path: Path) -> list[str]:
    """Export KEY=VALUE lines from a dotenv file into os.environ; returns the
    names set. Existing values win - never overrides what the shell gave us."""
    import os

    if not path.exists():
        return []
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().removeprefix("export ").strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip().strip("'\"")
            names.append(k)
    return names


def serve(
    store_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8484,
    open_browser: bool = True,
) -> None:
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        # fire after bind; a short timer is simpler than wiring a uvicorn
        # startup hook for a one-shot convenience
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(store_path), host=host, port=port)
