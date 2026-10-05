"""Run one RunConfig over a manifest: sample k per row, grade, persist.

Identity comes from config_fp — a rerun dedupes at (run_id, row_id,
sample_idx), so a crashed eval resumes by sampling only missing indices.
`forecast=True` does everything except create the sampling client.
"""

import asyncio
import importlib.metadata
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from flipbook.config import RunConfig, config_fp, run_id
from flipbook.graders import extracts_answer, grade, is_empty_response
from flipbook.lint import Finding, lint
from flipbook.manifest import Manifest
from flipbook.pricing import PRICES, estimate_usd
from flipbook.store import Store

# Gen-token estimate per effort for forecasting when no store prior exists.
# From the reference repo's observed means; historically off 5x+ on hard
# tasks, so the forecast labels which prior it used.
GEN_TOKEN_EST = {0.0: 150, 0.2: 400, 0.7: 1200, 0.9: 1500, 0.99: 2500}


@dataclass(frozen=True)
class Resolved:
    base_model: str
    renderer: str
    checkpoint_path: str | None


@dataclass(frozen=True)
class Forecast:
    cells: int  # sample slots not yet in the store
    prompt_tokens: int
    est_gen_tokens: int
    prior: str  # "store" or "table"
    usd_discount: float | None
    usd_list: float | None


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    manifest_hash: str
    n_rows: int
    n_new_samples: int
    n_errors: int
    pass1: float | None
    est_cost_usd: float | None
    forecast: Forecast | None = None


class LintFailed(Exception):
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        super().__init__("; ".join(f"{f.code}: {f.message}" for f in findings))


async def resolve_model(
    model: str,
    renderer: str | None = None,
    base_model: str | None = None,
    service_client: Any = None,
) -> Resolved:
    """tinker:// paths resolve base model + renderer via free REST metadata."""
    from tinker_cookbook import model_info

    if not model.startswith("tinker://"):
        return Resolved(model, renderer or model_info.get_recommended_renderer_name(model), None)

    from tinker_cookbook import checkpoint_utils

    if service_client is None and base_model is None:
        raise ValueError("tinker:// model needs API access to resolve its base model")
    if service_client is not None:
        #  explicit > checkpoint metadata > model default
        rest = service_client.create_rest_client()
        run = await rest.get_training_run_by_tinker_path_async(model)
        if base_model is not None and base_model != run.base_model:
            raise ValueError(
                f"base_model {base_model} does not match training run base {run.base_model}"
            )
        base_model = base_model or run.base_model
        if renderer is None:
            renderer = await checkpoint_utils.get_renderer_name_from_checkpoint_async(
                service_client, model
            )
    if renderer is None:
        renderer = model_info.get_recommended_renderer_name(base_model)
    return Resolved(base_model, renderer, model)


def _prompt_ints(renderer, renderer_name: str, messages: list[dict], effort: float | None) -> list[int]:
    # effort conditioning exists only on tml_v0; lint already gates misuse
    if renderer_name == "tml_v0":
        return renderer.build_generation_prompt(messages, effort=effort).to_ints()
    return renderer.build_generation_prompt(messages).to_ints()


def _failure_kind(
    verdict: float, extracted: str | None, stop_reason: str, text: str | None,
    grader_id: str | None = None,
) -> str | None:
    # precedence: a correct answer is never a failure, truncation only counts
    # when it caused the failure, and an empty message is not a parse failure
    if verdict == 1.0:
        return None
    if stop_reason == "length":
        return "truncation"
    if is_empty_response(text):
        return "empty"
    if extracted is None:
        # WHY the grader matters: for extraction graders (boxed, regex…) a
        # None means nothing parseable was produced; predicate and judge
        # graders legitimately return extracted=None on a plain wrong answer,
        # which is "failed" — calling it "parse" mislabels constraint
        # violations as unparseable output
        return "parse" if (grader_id is None or extracts_answer(grader_id)) else "failed"
    return "wrong_answer"


def _train_step(checkpoint_path: str | None) -> int | None:
    if not checkpoint_path:
        return None
    m = re.search(r"/(\d+)$", checkpoint_path)
    return int(m.group(1)) if m else None


def _prior_gen_mean(store: Store, manifest_hash: str, effort: float | None) -> tuple[float, list[str]] | None:
    """Mean gen_tokens over all stored runs on this manifest at the nearest effort."""
    runs = [r for r in store.runs() if r.get("manifest_hash") == manifest_hash]
    if not runs:
        return None
    near = min(abs((r.get("effort") or 0.0) - (effort or 0.0)) for r in runs)
    cands = [r for r in runs if abs((r.get("effort") or 0.0) - (effort or 0.0)) == near]
    gens = [
        s["gen_tokens"]
        for r in cands
        for s in store.samples(r["run_id"]).to_pylist()
        if s["gen_tokens"] is not None
    ]
    return (sum(gens) / len(gens), [r["run_id"] for r in cands]) if gens else None


async def evaluate_async(
    cfg: RunConfig,
    store: Store,
    *,
    forecast: bool = False,
    concurrency: int = 8,
    log: Callable[[str], None] = print,
    sampling_client: Any = None,
    base_model: str | None = None,
) -> RunSummary:
    import tinker
    from tinker.types import ModelInput, SamplingParams
    from tinker_cookbook.renderers import get_renderer, get_text_content
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    manifest = Manifest.load(store, cfg.manifest_hash)
    rows = manifest.rows
    rid = run_id(cfg)

    # resume bookkeeping before any client or tokenizer exists
    have: dict[str, set[int]] = {}
    for s in store.samples(rid).to_pylist():
        have.setdefault(s["row_id"], set()).add(s["sample_idx"])
    todo = {
        r["row_id"]: [i for i in range(cfg.k) if i not in have.get(r["row_id"], set())]
        for r in rows
    }
    cells = sum(len(v) for v in todo.values())

    if cells == 0 and not forecast:
        done = store.samples(rid).to_pylist()
        ok = [s for s in done if s["verdict"] is not None]
        return RunSummary(
            run_id=rid,
            manifest_hash=manifest.manifest_hash,
            n_rows=len(rows),
            n_new_samples=0,
            n_errors=sum(1 for s in done if s["failure_kind"] == "error"),
            pass1=sum(s["verdict"] for s in ok) / len(ok) if ok else None,
            est_cost_usd=sum(s["est_cost_usd"] or 0 for s in done) or None,
        )

    # ServiceClient resolves auth
    sc = None
    needs_rest = cfg.model.startswith("tinker://") and base_model is None
    if needs_rest or (not forecast and sampling_client is None):
        sc = tinker.ServiceClient()
    resolved = await resolve_model(
        cfg.model, cfg.renderer, base_model=base_model, service_client=sc
    )

    # lint before any paid client exists; grader ids resolve or the eval
    # would burn a paid run on rows that all land as grader_error
    findings = lint(
        cfg, resolved.base_model, grader_ids={r["grader_id"] for r in rows}
    )
    errors = [f for f in findings if f.level == "error"]
    if errors:
        raise LintFailed(errors)

    renderer = get_renderer(resolved.renderer, get_tokenizer(resolved.base_model))
    prompt_ints = {
        r["row_id"]: _prompt_ints(renderer, resolved.renderer, r["messages"], cfg.effort)
        for r in rows
    }
    tot_in = sum(len(prompt_ints[r]) for r, miss in todo.items() if miss)

    prior = _prior_gen_mean(store, manifest.manifest_hash, cfg.effort)
    est_gen = int(
        (prior[0] if prior else GEN_TOKEN_EST.get(round(cfg.effort or 0.0, 2), 1500)) * cells
    )
    disc = list_ = None
    if resolved.base_model in PRICES:
        disc = estimate_usd(resolved.base_model, "prefill", tot_in) + estimate_usd(
            resolved.base_model, "sample", est_gen
        )
        list_ = estimate_usd(resolved.base_model, "prefill", tot_in, list_price=True) + estimate_usd(
            resolved.base_model, "sample", est_gen, list_price=True
        )
    forecast_line = (
        f"forecast: {cells} cells · {tot_in:,} prompt tok · ~{est_gen:,} gen tok "
        f"(prior: {'store' if prior else 'table'})"
        + (
            f" · ~${disc:.3f} discount · ~${list_:.3f} list"
            if disc is not None
            else " · cost: unknown (no price table)"
        )
    )
    if forecast:
        return RunSummary(
            run_id=rid,
            manifest_hash=manifest.manifest_hash,
            n_rows=len(rows),
            n_new_samples=cells,
            n_errors=0,
            pass1=None,
            est_cost_usd=disc,
            forecast=Forecast(
                cells=cells,
                prompt_tokens=tot_in,
                est_gen_tokens=est_gen,
                prior="store" if prior else "table",
                usd_discount=disc,
                usd_list=list_,
            ),
        )

    # spend is always announced before the first paid call — the eval prints
    # this even when the caller didn't ask for a dry run
    log(forecast_line)

    # an injected client skips creating a new one, cfg.model still carries the path for identity and provenance
    client = sampling_client or (
        sc.create_sampling_client(model_path=resolved.checkpoint_path)
        if resolved.checkpoint_path
        else sc.create_sampling_client(base_model=resolved.base_model)
    )
    sem = asyncio.Semaphore(concurrency)

    async def work(row: dict) -> list[dict]:
        missing = todo[row["row_id"]]
        if not missing:
            return []
        toks = prompt_ints[row["row_id"]]
        async with sem:
            # why per-sample calls-seed binds the whole request, so num_samples=k under one seed returns identical draws
            async def one(i: int) -> dict:
                try:
                    resp = await client.sample_async(
                        ModelInput.from_ints(toks),
                        num_samples=1,
                        sampling_params=SamplingParams(
                            max_tokens=cfg.max_tokens,
                            temperature=cfg.temperature,
                            stop=renderer.get_stop_sequences(),
                            seed=cfg.seed + i,
                        ),
                    )
                except Exception as e:  # noqa: BLE001 — a failed cell is recorded, not fatal
                    return {
                        "run_id": rid, "row_id": row["row_id"], "sample_idx": i,
                        "prompt_tokens": len(toks), "error": f"{type(e).__name__}: {e}",
                        "failure_kind": "error",
                    }
                seq = resp.sequences[0]
                msg, _term = renderer.parse_response(seq.tokens)
                text = get_text_content(msg)
                try:
                    g = grade(row["grader_id"], text, row["gold"])
                    verdict, extracted, note = g.verdict, g.extracted, g.note
                except Exception as ge:  # noqa: BLE001 — a grader crash is a wrong answer
                    verdict, extracted, note = 0.0, None, f"grader_error: {type(ge).__name__}: {ge}"
                est = None
                if resolved.base_model in PRICES:
                    est = estimate_usd(resolved.base_model, "prefill", len(toks)) + estimate_usd(
                        resolved.base_model, "sample", len(seq.tokens)
                    )
                return {
                    "run_id": rid, "row_id": row["row_id"], "sample_idx": i,
                    "text": text, "prompt_tokens": len(toks), "gen_tokens": len(seq.tokens),
                    "stop_reason": seq.stop_reason, "verdict": verdict,
                    "extracted": extracted, "failure_kind": _failure_kind(verdict, extracted, seq.stop_reason, text, row["grader_id"]),
                    "grade_note": note, "error": None, "est_cost_usd": est,
                    "token_ids": list(seq.tokens), "token_logprobs": list(seq.logprobs) if seq.logprobs else None,
                }

            return await asyncio.gather(*(one(i) for i in missing))

    new_rows = []
    for done_n, coro in enumerate(asyncio.as_completed([work(r) for r in rows]), start=1):
        new_rows.extend(await coro)
        if done_n % 25 == 0 or done_n == len(rows):
            log(f"  {done_n}/{len(rows)} rows sampled")
    store.put_samples(rid, new_rows)

    cb_ver = importlib.metadata.version("tinker-cookbook")
    renderer_version = (
        f"tml-renderers=={importlib.metadata.version('tml-renderers')}"
        if resolved.renderer == "tml_v0"
        else f"tinker-cookbook=={cb_ver}"
    )
    store.put_run(
        {
            "run_id": rid, "config_fp": config_fp(cfg), "created_at": datetime.now(UTC).isoformat(),
            "manifest_hash": manifest.manifest_hash, "study": cfg.study, "source": "eval",
            # unlabeled runs still get an addressable name: tail of
            # "org/Model" or "tinker://…/weights_name" (Store.resolve_run)
            "label": cfg.label or cfg.model.rsplit("/", 1)[-1],
            "model_id": resolved.base_model,
            "checkpoint_path": resolved.checkpoint_path,
            "base_model": resolved.base_model if resolved.checkpoint_path else None,
            "train_step": _train_step(resolved.checkpoint_path),
            "renderer": resolved.renderer, "renderer_version": renderer_version,
            "effort": cfg.effort, "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens, "k": cfg.k, "seed": cfg.seed,
            "grader_id": "benchmark-native",
            "grader_version": f"tinker-cookbook=={cb_ver}",
            "sdk_versions": {
                "tinker": importlib.metadata.version("tinker"),
                "tinker-cookbook": cb_ver,
                "tml-renderers": importlib.metadata.version("tml-renderers"),
            },
            "provenance": {"evaluated_cells": cells},
        }
    )

    done = store.samples(rid).to_pylist()
    ok = [s for s in done if s["verdict"] is not None]
    return RunSummary(
        run_id=rid,
        manifest_hash=manifest.manifest_hash,
        n_rows=len(rows),
        n_new_samples=cells,
        n_errors=sum(1 for s in done if s["failure_kind"] == "error"),
        pass1=sum(s["verdict"] for s in ok) / len(ok) if ok else None,
        est_cost_usd=sum(s["est_cost_usd"] or 0 for s in done) or None,
    )


def evaluate(cfg: RunConfig, store: Store, *, forecast: bool = False, concurrency: int = 8) -> RunSummary:
    return asyncio.run(evaluate_async(cfg, store, forecast=forecast, concurrency=concurrency))
