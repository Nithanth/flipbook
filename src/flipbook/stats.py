"""Paired per-row statistics between two runs.

A row counts only when both runs have k non-error samples, and every
excluded row is named with a reason. 
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from flipbook.store import Store

BOOT_DRAWS = 10_000
FLIP_DELTA = 0.5


@dataclass(frozen=True)
class PairReport:
    run_a: str
    run_b: str
    n_pairs: int
    pairs: list[str]  # complete-pair row_ids, sorted
    excluded: list[dict]  # {row_id, reason}
    acc_a: float
    acc_b: float
    delta: float
    delta_ci: tuple[float, float]
    agreement: dict  # both_right / both_wrong / a_only / b_only
    flips: list[dict]  # {row_id, p_a, p_b, kind, hard}
    tokens: dict  # per-run {mean, p50, p90, p99}
    delta_tokens: float
    delta_tokens_ci: tuple[float, float]
    truncation_rate_a: float
    truncation_rate_b: float
    cost_a: float
    cost_b: float
    failures: dict  # per-run failure_kind counts over non-correct samples

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q)) if len(v) else 0.0


def _boot_ci(d: np.ndarray) -> tuple[float, float]:
    if not len(d):
        return (0.0, 0.0)
    rng = np.random.default_rng(0)  # fixed seed
    means = d[rng.integers(0, len(d), size=(BOOT_DRAWS, len(d)))].mean(axis=1)
    return _pct(means, 2.5), _pct(means, 97.5)


def _per_row(samples: list[dict], k: int) -> tuple[dict[str, dict], dict[str, str]]:
    """Group samples by row; complete rows need k non-error samples."""
    by_row: dict[str, list[dict]] = {}
    for s in samples:
        by_row.setdefault(s["row_id"], []).append(s)
    rows, excl = {}, {}
    for rid, ss in by_row.items():
        ok = [s for s in ss if s["verdict"] is not None and s["failure_kind"] != "error"]
        if len(ok) < k:
            excl[rid] = "errors" if any(s["failure_kind"] == "error" for s in ss) else "incomplete"
            continue
        rows[rid] = {
            "p": sum(s["verdict"] for s in ok) / len(ok),
            "gen": [s["gen_tokens"] or 0 for s in ok],
            "trunc": sum(s["stop_reason"] == "length" for s in ok) / len(ok),
        }
    return rows, excl


def _tok_stats(samples: list[dict]) -> dict:
    g = np.array([s["gen_tokens"] for s in samples if s["gen_tokens"] is not None], dtype=float)
    return {
        "mean": float(g.mean()) if len(g) else 0.0,
        "p50": _pct(g, 50), "p90": _pct(g, 90), "p99": _pct(g, 99),
    }


def compare(store: Store, run_a: str, run_b: str) -> PairReport:
    sa = store.samples(run_a).to_pylist()
    sb = store.samples(run_b).to_pylist()
    runs = {r["run_id"]: r for r in store.runs()}
    ka = runs.get(run_a, {}).get("k") or max((s["sample_idx"] for s in sa), default=-1) + 1
    kb = runs.get(run_b, {}).get("k") or max((s["sample_idx"] for s in sb), default=-1) + 1

    pa, ex_a = _per_row(sa, ka)
    pb, ex_b = _per_row(sb, kb)
    common = sorted(pa.keys() & pb.keys())
    # sym-diff misses rows excluded from BOTH runs (in neither pa nor pb) 
    excluded = [
        {"row_id": r, "reason": (ex_b.get(r, "missing_in_b") if r in pa else ex_a.get(r, "missing_in_a"))}
        for r in sorted(set(pa) ^ set(pb))
    ]
    excluded += [
        {"row_id": r, "reason": f"{ex_a[r]}_in_both"} for r in sorted(set(ex_a) & set(ex_b))
    ]

    pa_v = np.array([pa[r]["p"] for r in common])
    pb_v = np.array([pb[r]["p"] for r in common])
    d = pb_v - pa_v
    ga = np.array([np.mean(pa[r]["gen"]) for r in common])
    gb = np.array([np.mean(pb[r]["gen"]) for r in common])

    ma, mb = pa_v >= 0.5, pb_v >= 0.5  # majority verdict: p_r >= 0.5
    agreement = {
        "both_right": int((ma & mb).sum()), "both_wrong": int((~ma & ~mb).sum()),
        "a_only": int((ma & ~mb).sum()), "b_only": int((~ma & mb).sum()),
    }
    flips = [
        {
            "row_id": r, "p_a": float(x), "p_b": float(y),
            "kind": "regression" if x - y >= FLIP_DELTA else "gain",
            "hard": (x, y) in ((1.0, 0.0), (0.0, 1.0)),
        }
        for r, x, y in zip(common, pa_v, pb_v)
        if abs(float(y) - float(x)) >= FLIP_DELTA
    ]

    fails: dict[str, dict[str, int]] = {"a": {}, "b": {}}
    for key, ss in (("a", sa), ("b", sb)):
        for s in ss:
            if s["failure_kind"]:
                fails[key][s["failure_kind"]] = fails[key].get(s["failure_kind"], 0) + 1

    n_a = len(sa) or 1
    n_b = len(sb) or 1
    return PairReport(
        run_a=run_a, run_b=run_b, n_pairs=len(common), pairs=common, excluded=excluded,
        acc_a=float(pa_v.mean()) if len(pa_v) else 0.0,
        acc_b=float(pb_v.mean()) if len(pb_v) else 0.0,
        delta=float(d.mean()) if len(d) else 0.0,
        delta_ci=_boot_ci(d),
        agreement=agreement, flips=flips,
        tokens={"a": _tok_stats(sa), "b": _tok_stats(sb)},
        delta_tokens=float((gb - ga).mean()) if len(common) else 0.0,
        delta_tokens_ci=_boot_ci(gb - ga),
        truncation_rate_a=sum(s["stop_reason"] == "length" for s in sa) / n_a,
        truncation_rate_b=sum(s["stop_reason"] == "length" for s in sb) / n_b,
        cost_a=sum(s["est_cost_usd"] or 0 for s in sa),
        cost_b=sum(s["est_cost_usd"] or 0 for s in sb),
        failures=fails,
    )


def gate(
    pair: PairReport,
    max_regression: float = 0.02,
    max_new_truncation_rate: float = 0.0,
) -> tuple[bool, list[str]]:
    """CI-aware regression gate: fails only when the CI excludes zero."""
    reasons = []
    if pair.delta < -max_regression and pair.delta_ci[1] < 0:
        reasons.append(
            f"delta {pair.delta:+.3f} < -{max_regression} with CI hi {pair.delta_ci[1]:+.3f} < 0"
        )
    dt = pair.truncation_rate_b - pair.truncation_rate_a
    if dt > max_new_truncation_rate:
        reasons.append(f"new truncations {dt:+.3f} > {max_new_truncation_rate}")
    return (not reasons, reasons)
