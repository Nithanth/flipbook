"""Per-run token and cost distribution - sizing tool for `max_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass

from flipbook.store import Store


@dataclass(frozen=True)
class BudgetReport:
    run_id: str
    n_samples: int # total ros in samples/{run_id}
    n_truncated: int # amt of responses truncated - max tokens hit
    gen_p50: float # median tokens generated
    gen_p95: float # 95th percentile tokens generated
    gen_p99: float # 99th percentile tokens generated
    gen_max: int # longest generated sequence
    total_prompt_tokens: int # prefill spend in tokens
    total_gen_tokens: int # sample spend in tokens
    est_cost_usd: float | None # sum of per-sample est_cost_usd or none if unpriced


def _pct(xs: list[int], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return float(xs[min(len(xs) - 1, int(q / 100 * len(xs)))])


def budget(store: Store, run_id: str) -> BudgetReport:
    tbl = store.samples(run_id)
    rows = tbl.to_pylist()
    gens = [r["gen_tokens"] for r in rows if r["gen_tokens"] is not None]
    costs = [r["est_cost_usd"] for r in rows if r["est_cost_usd"] is not None]
    return BudgetReport(
        run_id=run_id,
        n_samples=tbl.num_rows,
        n_truncated=sum(1 for r in rows if r["failure_kind"] == "truncation"),
        gen_p50=_pct(gens, 50),
        gen_p95=_pct(gens, 95),
        gen_p99=_pct(gens, 99),
        gen_max=max(gens, default=0),
        total_prompt_tokens=sum(r["prompt_tokens"] or 0 for r in rows),
        total_gen_tokens=sum(gens),
        est_cost_usd=sum(costs) if costs else None,
    )
