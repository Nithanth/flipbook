"""Per-token logprob gap: score each baseline trace under both checkpoints.

delta[i] = lp_ckpt(t_i) - lp_base(t_i) over the baseline's own sampled
tokens — "how surprised is the new checkpoint by what the old model did."
Both lp arrays come from compute_logprobs on the same P ++ t sequence —
never sampling-time logprobs — so positions align exactly. p_skip scores
P(eom | P): the direct "did it stop thinking" read.

Per row: P is rendered at the BASE run's effort, t is the base run's
lowest-idx clean sample, and 4 calls score (P+t under base, P+t under ckpt,
P+eom under base, P+eom under ckpt) — all prefill-priced.
"""


import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flipbook.pricing import PRICES, estimate_usd
from flipbook.runner import _prompt_ints
from flipbook.stats import comparability, compare
from flipbook.store import Store

TAU = 5.0 
WIN = 16  # window for the smoothed argmin


@dataclass(frozen=True)
class DivergenceSummary:
    base_run_id: str
    ckpt_run_id: str
    n_rows: int
    mean_sum_nats: float
    mean_p_skip_base: float
    mean_p_skip_ckpt: float
    n_diverged: int  # rows whose cumulative gap crossed -tau
    est_cost_usd: float | None
    forecast: dict | None = None


def _run_rec(store: Store, run_id: str) -> dict:
    for r in store.runs():
        if r["run_id"] == run_id or r["run_id"].startswith(run_id):
            return r
    raise SystemExit(f"no run {run_id!r} in store")


def _pos(delta: list[float], tau: float) -> int | None:
    cum = 0.0
    for i, d in enumerate(delta):
        cum += d
        if cum <= -tau:
            return i
    return None


def _win_argmin(delta: list[float], w: int = WIN) -> int | None:
    if len(delta) < w:
        return None
    means = [sum(delta[i : i + w]) / w for i in range(len(delta) - w + 1)]
    return means.index(min(means)) + w // 2  # report the window's center


def _pick_trace(samples: list[dict]) -> dict | None:
    """Lowest-idx non-error sample that kept its token ids."""
    ok = [s for s in samples if s["failure_kind"] != "error" and s["token_ids"]]
    return min(ok, key=lambda s: s["sample_idx"], default=None)


async def diverge_async(
    store: Store,
    base_run_id: str,
    ckpt_run_id: str,
    rows: list[str] | None = None,
    *,
    tau: float = TAU,
    forecast: bool = False,
    concurrency: int = 8,
    log: Callable[[str], None] = print,
) -> DivergenceSummary:
    import tinker
    from tinker.types import ModelInput
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    base = _run_rec(store, base_run_id)
    ckpt = _run_rec(store, ckpt_run_id)
    if base["manifest_hash"] != ckpt["manifest_hash"]:
        raise SystemExit("base and ckpt runs used different manifests")
    if not comparability(base, ckpt)["token_views"]:
        raise ValueError(
            "per-token divergence requires a shared renderer/vocabulary: "
            f"{base.get('renderer')} vs {ckpt.get('renderer')}"
        )

    # complete pairs only — scoring a row only one run judged means nothing
    pair = compare(store, base["run_id"], ckpt["run_id"])
    row_ids = sorted(rows) if rows is not None else pair.pairs
    mrows = {r["row_id"]: r for r in store.manifest_rows(base["manifest_hash"])}
    by_row: dict[str, list[dict]] = {}
    for s in store.samples(base["run_id"]).to_pylist():
        by_row.setdefault(s["row_id"], []).append(s)

    done = {(r["row_id"], r["sample_idx"]) for r in store.divergence(base["run_id"], ckpt["run_id"]).to_pylist()}
    renderer = get_renderer(base["renderer"], get_tokenizer(base["model_id"]))
    # P rendered at the baseline's effort — the trace was generated under it
    prompts = {
        rid: _prompt_ints(renderer, base["renderer"], mrows[rid]["messages"], base["effort"])
        for rid in row_ids
    }
    traces = {rid: _pick_trace(by_row.get(rid, [])) for rid in row_ids}
    todo = {
        rid: t for rid, t in traces.items()
        if t is not None and (rid, t["sample_idx"]) not in done
    }

    # billed tokens per row: 2 trace scorings (P+t) + 2 p_skip scorings (P+eom)
    prefill_per_row = {
        rid: 2 * (len(prompts[rid]) + len(t["token_ids"])) + 2 * (len(prompts[rid]) + 1)
        for rid, t in todo.items()
    }
    tot = sum(prefill_per_row.values())
    if forecast or not todo:
        disc = estimate_usd(base["model_id"], "prefill", tot) if base["model_id"] in PRICES else None
        lst = estimate_usd(base["model_id"], "prefill", tot, list_price=True) if base["model_id"] in PRICES else None
        return DivergenceSummary(
            base["run_id"], ckpt["run_id"], len(todo), 0.0, 0.0, 0.0, 0,
            disc, {"prefill_tokens": tot, "usd_list": lst} if forecast else None,
        )

    sc = tinker.ServiceClient()

    def _client(rec: dict) -> Any:
        return (
            sc.create_sampling_client(model_path=rec["checkpoint_path"])
            if rec["checkpoint_path"]
            else sc.create_sampling_client(base_model=rec["model_id"])
        )

    c_base, c_ckpt = _client(base), _client(ckpt)
    stops = list(renderer.get_stop_sequences())
    sem = asyncio.Semaphore(concurrency)

    async def _cont_lp(client, seq: list[int], plen: int) -> list[float]:
        # lps[i] = log P(x_i | x_<i); the continuation starts at index plen
        lps = await client.compute_logprobs_async(ModelInput.from_ints(seq))
        return [float(x) for x in lps[plen:]]

    async def _p_skip(client, p: list[int]) -> float:
        # sum P(stop_tok first) over stop ids — exact for single-stop renderers
        tot = 0.0
        for st in stops:
            lps = await client.compute_logprobs_async(ModelInput.from_ints(p + [st]))
            tot += math.exp(lps[len(p)])
        return tot

    async def work(rid: str) -> dict | None:
        t = traces[rid]
        p, tt = prompts[rid], t["token_ids"]
        try:
            async with sem:
                lb, lc = await asyncio.gather(
                    _cont_lp(c_base, p + tt, len(p)), _cont_lp(c_ckpt, p + tt, len(p))
                )
                sb, sk = await asyncio.gather(_p_skip(c_base, p), _p_skip(c_ckpt, p))
        except Exception as e:  # noqa: BLE001 — one bad row shouldn't kill the run
            log(f"  {rid}: {type(e).__name__}: {e}")
            return None
        delta = [c - b for b, c in zip(lb, lc)]
        cost = None
        if base["model_id"] in PRICES:
            cost = estimate_usd(base["model_id"], "prefill", prefill_per_row[rid])
        return {
            "base_run_id": base["run_id"], "ckpt_run_id": ckpt["run_id"],
            "row_id": rid, "sample_idx": t["sample_idx"], "prompt_len": len(p), "n": len(tt),
            "lp_base": lb, "lp_ckpt": lc, "delta": delta,
            "sum_nats": sum(delta), "mean_nats": sum(delta) / len(delta),
            "divergence_pos": _pos(delta, tau), "win_argmin": _win_argmin(delta),
            "p_skip_base": sb, "p_skip_ckpt": sk, "cost_usd": cost,
        }

    out = []
    for done_n, coro in enumerate(asyncio.as_completed([work(r) for r in todo]), start=1):
        if (r := await coro) is not None:
            out.append(r)
        if done_n % 25 == 0 or done_n == len(todo):
            log(f"  {done_n}/{len(todo)} rows scored")
    store.put_divergence(base["run_id"], ckpt["run_id"], out)

    all_rows = store.divergence(base["run_id"], ckpt["run_id"]).to_pylist()
    n = len(all_rows) or 1
    return DivergenceSummary(
        base["run_id"], ckpt["run_id"], len(all_rows),
        sum(r["sum_nats"] for r in all_rows) / n,
        sum(r["p_skip_base"] for r in all_rows) / n,
        sum(r["p_skip_ckpt"] for r in all_rows) / n,
        sum(r["divergence_pos"] is not None for r in all_rows),
        sum(r["cost_usd"] or 0 for r in all_rows) or None,
    )


async def branch(
    store: Store,
    base_run_id: str,
    ckpt_run_id: str,
    row_id: str,
    sample_idx: int,
    pos: int,
    n: int = 96,
) -> dict:
    """Counterfactual continuation: greedy ckpt sample from a cut on the base trace.

    The per-token delta says *where* the ckpt disagrees but never *what it would
    do instead* — this answers the second question. Uses the same prefix
    convention as diverge_async (base-effort prompt ++ base's own tokens[:pos])
    so the ckpt continuation is conditioned on the reasoning it was scored
    against. pos indexes continuation tokens.
    """
    import tinker
    from tinker.types import ModelInput, SamplingParams
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from flipbook.decode import strip_control_tokens

    base = _run_rec(store, base_run_id)
    ckpt = _run_rec(store, ckpt_run_id)
    key = (row_id, int(sample_idx))
    drow = next(
        (
            r
            for r in store.divergence(base["run_id"], ckpt["run_id"]).to_pylist()
            if (r["row_id"], r["sample_idx"]) == key
        ),
        None,
    )
    srow = next(
        (
            r
            for r in store.samples(base["run_id"]).to_pylist()
            if (r["row_id"], r["sample_idx"]) == key
        ),
        None,
    )
    mrow = next(
        (r for r in store.manifest_rows(base["manifest_hash"]) if r["row_id"] == row_id),
        None,
    )
    if drow is None or srow is None or not srow.get("token_ids") or mrow is None:
        raise LookupError(f"no divergence/sample data for {row_id}:{sample_idx}")
    ids = [int(i) for i in srow["token_ids"]]
    pos = max(0, min(int(pos), len(ids) - 1))
    n = max(1, min(int(n), 256))

    tok = get_tokenizer(base["model_id"])
    renderer = get_renderer(base["renderer"], tok)
    prompt = _prompt_ints(renderer, base["renderer"], mrow["messages"], base["effort"])

    sc = tinker.ServiceClient()
    client = (
        sc.create_sampling_client(model_path=ckpt["checkpoint_path"])
        if ckpt["checkpoint_path"]
        else sc.create_sampling_client(base_model=ckpt["model_id"])
    )
    resp = await client.sample_async(
        ModelInput.from_ints(prompt + ids[:pos]),
        num_samples=1,
        sampling_params=SamplingParams(
            max_tokens=n,
            temperature=0.0,
            stop=renderer.get_stop_sequences(),
            seed=0,
        ),
    )
    seq = resp.sequences[0]
    ckpt_ids = list(seq.tokens)
    ckpt_lps = list(seq.logprobs) if seq.logprobs else []
    return {
        "pos": pos,
        "base_tok": tok.decode([ids[pos]]),
        "base_tok_lp": float(drow["lp_base"][pos]),
        "ckpt_lp_on_base_tok": float(drow["lp_ckpt"][pos]),
        "ckpt_first_tok": tok.decode([ckpt_ids[0]]) if ckpt_ids else "",
        "ckpt_first_tok_lp": float(ckpt_lps[0]) if ckpt_lps else None,
        "base_cont": strip_control_tokens(tok.decode(ids[pos : pos + n])),
        "ckpt_cont": strip_control_tokens(tok.decode(ckpt_ids)),
    }


def diverge(
    store: Store,
    base_run_id: str,
    ckpt_run_id: str,
    rows: list[str] | None = None,
    **kw,
) -> DivergenceSummary:
    return asyncio.run(diverge_async(store, base_run_id, ckpt_run_id, rows, **kw))
