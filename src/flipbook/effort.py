"""Effort-gap: does the effort prefix still change what the model believes?

For each stored trace t (sampled at e_high), score it under the same model
with the prompt rendered at e_high vs e_low. gap = lp_high - lp_low -> a ratio vs the base
model's mean gap (computed by the caller) reads 1 = intact, -> 0 = the
prefix stopped mattering. This measures the prefix effect, which is the
deployed effort mechanism.
"""


import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flipbook.diverge import _run_rec
from flipbook.pricing import PRICES, estimate_usd
from flipbook.runner import _prompt_ints
from flipbook.store import Store


@dataclass(frozen=True)
class EffortSummary:
    run_id: str
    e_low: float
    e_high: float
    n_rows: int
    mean_gap_nats: float
    est_cost_usd: float | None
    forecast: dict | None = None


async def effort_gap_async(
    store: Store,
    run_id: str,
    e_low: float,
    e_high: float,
    rows: list[str] | None = None,
    *,
    forecast: bool = False,
    concurrency: int = 8,
    log: Callable[[str], None] = print,
) -> EffortSummary:
    import tinker
    from tinker.types import ModelInput
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    rec = _run_rec(store, run_id)
    if rec["renderer"] != "tml_v0":
        raise SystemExit("effort gap needs an effort-conditioned renderer (tml_v0)")
    # t must be a trace this run actually produced under the e_high prefix 
    if abs((rec["effort"] or 0.0) - e_high) > 1e-6:
        raise SystemExit(f"run was sampled at effort {rec['effort']}, not e_high {e_high}")

    mrows = {r["row_id"]: r for r in store.manifest_rows(rec["manifest_hash"])}
    samples = [
        s for s in store.samples(rec["run_id"]).to_pylist()
        if s["failure_kind"] != "error" and s["token_ids"]
        and (rows is None or s["row_id"] in rows)
    ]
    done = {
        (r["row_id"], r["sample_idx"])
        for r in store.effort(rec["run_id"], e_low, e_high).to_pylist()
    }
    todo = [s for s in samples if (s["row_id"], s["sample_idx"]) not in done]

    renderer = get_renderer(rec["renderer"], get_tokenizer(rec["model_id"]))
    # billed per trace: t scored under both prefixes (lo + t, hi + t)
    tok_per = {
        (s["row_id"], s["sample_idx"]): 2 * len(s["token_ids"]) + len(
            _prompt_ints(renderer, "tml_v0", mrows[s["row_id"]]["messages"], e_low)
        ) + len(
            _prompt_ints(renderer, "tml_v0", mrows[s["row_id"]]["messages"], e_high)
        )
        for s in todo
    }
    tot = sum(tok_per.values())
    if forecast or not todo:
        disc = estimate_usd(rec["model_id"], "prefill", tot) if rec["model_id"] in PRICES else None
        lst = estimate_usd(rec["model_id"], "prefill", tot, list_price=True) if rec["model_id"] in PRICES else None
        return EffortSummary(
            rec["run_id"], e_low, e_high, len(todo), 0.0, disc,
            {"prefill_tokens": tot, "usd_list": lst} if forecast else None,
        )

    sc = tinker.ServiceClient()
    client: Any = (
        sc.create_sampling_client(model_path=rec["checkpoint_path"])
        if rec["checkpoint_path"]
        else sc.create_sampling_client(base_model=rec["model_id"])
    )
    sem = asyncio.Semaphore(concurrency)

    async def _lp_sum(seq: list[int], plen: int) -> float:
        # lps[i] = log P(x_i | x_<i); sum the continuation only
        lps = await client.compute_logprobs_async(ModelInput.from_ints(seq))
        return sum(x for x in lps[plen:] if x is not None)

    async def work(s: dict) -> dict | None:
        tt = s["token_ids"]
        msgs = mrows[s["row_id"]]["messages"]
        try:
            async with sem:
                lo = _prompt_ints(renderer, "tml_v0", msgs, e_low)
                hi = _prompt_ints(renderer, "tml_v0", msgs, e_high)
                l_lo, l_hi = await asyncio.gather(
                    _lp_sum(lo + tt, len(lo)), _lp_sum(hi + tt, len(hi))
                )
        except Exception as e:  # noqa: BLE001 — one bad row shouldn't kill the run
            log(f"  {s['row_id']}:{s['sample_idx']}: {type(e).__name__}: {e}")
            return None
        cost = None
        if rec["model_id"] in PRICES:
            cost = estimate_usd(rec["model_id"], "prefill", tok_per[(s["row_id"], s["sample_idx"])])
        return {
            "run_id": rec["run_id"], "row_id": s["row_id"], "sample_idx": s["sample_idx"],
            "e_low": e_low, "e_high": e_high,
            "lp_low_sum": l_lo, "lp_high_sum": l_hi, "gap_nats": l_hi - l_lo,
            "n": len(tt), "cost_usd": cost,
        }

    out = []
    for done_n, coro in enumerate(asyncio.as_completed([work(s) for s in todo]), start=1):
        if (r := await coro) is not None:
            out.append(r)
        if done_n % 25 == 0 or done_n == len(todo):
            log(f"  {done_n}/{len(todo)} traces scored")
    store.put_effort(rec["run_id"], e_low, e_high, out)

    all_rows = store.effort(rec["run_id"], e_low, e_high).to_pylist()
    n = len(all_rows) or 1
    return EffortSummary(
        rec["run_id"], e_low, e_high, len(all_rows),
        sum(r["gap_nats"] for r in all_rows) / n,
        sum(r["cost_usd"] or 0 for r in all_rows) or None,
    )


def effort_gap(store: Store, run_id: str, e_low: float, e_high: float, rows=None, **kw) -> EffortSummary:
    return asyncio.run(effort_gap_async(store, run_id, e_low, e_high, rows, **kw))
