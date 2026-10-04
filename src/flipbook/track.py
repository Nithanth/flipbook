"""Post-hoc tracking: evaluate every saved checkpoint of a finished run.
Reads a cookbook log dir's checkpoints.jsonl (through TrainingRunStore, not
raw jsonl) and runs the same pipeline the in-loop evaluator would have
"""


import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from flipbook.config import RunConfig, run_id
from flipbook.manifest import Manifest
from flipbook.runner import evaluate_async
from flipbook.stats import compare
from flipbook.store import Store


@dataclass(frozen=True)
class TrackPlan:
    log_dir: str
    checkpoints: list[dict]  # {batch, sampler_path, run_id}
    base_run_id: str | None
    total_cells: int
    est_cost_usd: float | None


def _select(records: list[dict], every: int | None, last: int | None) -> list[dict]:
    recs = [r for r in records if r.get("sampler_path")]
    if every:
        recs = [r for r in recs if (r.get("batch") or 0) % every == 0]
    if last:
        recs = recs[-last:]
    return recs


def _checkpoints(log_dir: str | Path) -> list[dict]:
    from tinker_cookbook.stores.storage import LocalStorage
    from tinker_cookbook.stores.training_store import TrainingRunStore
    store = TrainingRunStore(LocalStorage(str(log_dir)))
    return store.read_checkpoints()


def _cfg_for(
    model: str, manifest: Manifest, renderer: str, *,
    effort, k, temperature, max_tokens, seed, study, label=None
) -> RunConfig:
    return RunConfig(
        model=model, manifest_hash=manifest.manifest_hash, effort=effort,
        temperature=temperature, max_tokens=max_tokens, k=k, seed=seed,
        renderer=renderer, study=study, label=label,
    )


def _stamp(store: Store, rid: str, step: int | None, model_path: str) -> None:
    """The run record knows its weights path; add the measured train step."""
    f = store.path / "runs" / f"{rid}.json"
    rec = json.loads(f.read_text())
    prov = rec.setdefault("provenance", {})
    prov["model_path"] = model_path
    if step is not None:
        prov["train_step_measured"] = step
    f.write_text(json.dumps(rec, indent=2))


async def track_async(
    store: Store,
    log_dir: str | Path,
    manifest: str,
    *,
    every: int | None = None,
    last: int | None = None,
    include_base: bool = False,
    base_model: str | None = None,
    diverge: bool = False,
    effort_pair: tuple[float, float] | None = None,
    effort: float = 0.9,
    k: int = 4,
    temperature: float = 0.6,
    max_tokens: int = 32768,
    seed: int = 0,
    study: str | None = None,
    forecast: bool = False,
    concurrency: int = 8,
    log: Callable[[str], None] = print,
) -> TrackPlan:
    study = study or Path(log_dir).name
    m = Manifest.load(store, manifest)
    ckpts = _select(_checkpoints(log_dir), every, last)
    if not ckpts:
        raise SystemExit(f"no checkpoints with sampler_path under {log_dir}")
    # renderer is part of RunConfig identity
    from flipbook.runner import resolve_model
    sc = None
    if base_model is None:
        import tinker
        sc = tinker.ServiceClient()
    resolved = await resolve_model(
        ckpts[0]["sampler_path"], base_model=base_model, service_client=sc
    )
    renderer = resolved.renderer
    kw = {"effort": effort, "k": k, "temperature": temperature, "max_tokens": max_tokens,
          "seed": seed, "study": study}
    total_cells = 0
    total_cost = 0.0
    priced = True
    base_rid = None
    if include_base:
        bcfg = _cfg_for(resolved.base_model, m, renderer, **kw)
        base_rid = run_id(bcfg)
        s = await evaluate_async(bcfg, store, forecast=forecast, concurrency=concurrency,
                                 log=log, base_model=base_model)
        total_cells += s.forecast.cells if s.forecast else s.n_new_samples
        if s.est_cost_usd is None:
            priced = False
        else:
            total_cost += s.est_cost_usd
        if not forecast:
            log(f"base {base_rid}: pass@1={s.pass1}")
    plan_ckpts = []
    for c in ckpts:
        # WHY name over batch: cookbook writes batch=0 for the "final" record,
        # while numeric names ("000056") are zero-padded batch numbers
        name = c.get("name") or ""
        step = int(name) if name.isdigit() else None
        cfg = _cfg_for(c["sampler_path"], m, renderer,
                       label=name or (f"step{step}" if step is not None else None),
                       **kw)
        s = await evaluate_async(cfg, store, forecast=forecast, concurrency=concurrency,
                                 log=log, base_model=base_model)
        total_cells += s.forecast.cells if s.forecast else s.n_new_samples
        if s.est_cost_usd is None:
            priced = False
        else:
            total_cost += s.est_cost_usd
        plan_ckpts.append({"batch": c.get("batch"), "sampler_path": c["sampler_path"],
                           "run_id": run_id(cfg)})
        if not forecast:
            _stamp(store, run_id(cfg), step, c["sampler_path"])
            log(f"{name or 'ckpt'} {run_id(cfg)}: pass@1={s.pass1} ~${s.est_cost_usd or 0:.3f}")
            if diverge and base_rid:
                from flipbook.diverge import diverge_async
                pair = compare(store, base_rid, run_id(cfg))
                flips = [f["row_id"] for f in pair.flips]
                ds = await diverge_async(store, base_rid, run_id(cfg), flips or None,
                                         concurrency=concurrency, log=log)
                log(f"  diverge: {ds.n_rows} rows, mean gap {ds.mean_sum_nats:+.1f} nats")
            if effort_pair and abs(effort - effort_pair[1]) < 1e-6:
                from flipbook.effort import effort_gap_async
                es = await effort_gap_async(store, run_id(cfg), *effort_pair,
                                            concurrency=concurrency, log=log)
                log(f"  effort gap {es.mean_gap_nats:+.1f} nats over {es.n_rows} traces")
    return TrackPlan(
        str(log_dir), plan_ckpts, base_rid, total_cells,
        total_cost if priced else None,
    )


def track(store: Store, log_dir, manifest: str, **kw) -> TrackPlan:
    return asyncio.run(track_async(store, log_dir, manifest, **kw))
