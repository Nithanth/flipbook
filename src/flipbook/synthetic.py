"""Synthetic demo data - writes a fake-but-plausible training collapse into
the store through the same put_* APIs a real eval uses, so every surface
(status, compare, study charts, divergence view) works with zero spend and
zero credentials. Everything is stamped study="synthetic" and run labels
make the invented provenance obvious - these numbers illustrate the tool's
workflow, they are not measurements.
"""
from __future__ import annotations

import math
import random
from datetime import UTC, datetime

from flipbook.manifest import Manifest
from flipbook.store import Store

STUDY = "synthetic"
MANIFEST_NAME = "synthetic_demo"

# small invented set - enough rows for mean@k, flips, and CIs to look real
_QUESTIONS = [
    ("Nina buys 3 notebooks at $4 each and pays with a $20 bill. How much change does she get?", "8"),
    ("A train travels 180 km in 2 hours. What is its average speed in km/h?", "90"),
    ("If 5x - 7 = 18, what is x?", "5"),
    ("A rectangle has perimeter 30 and width 4. What is its area?", "44"),
    ("Priya scored 80, 90, and 70 on three tests. What is her average?", "80"),
    ("A jar has 12 red and 8 blue marbles. What fraction are red? Answer as a decimal.", "0.6"),
    ("What is 15% of 240?", "36"),
    ("Tom is twice as old as Sam. Together their ages sum to 36. How old is Sam?", "12"),
    ("A tank holds 50 L and loses 3 L per hour. How many litres remain after 9 hours?", "23"),
    ("Compute (3 + 4) * (8 - 2) - 7.", "35"),
]

# the collapse story: (label, train_step, acc@2-samples, p_skip, gen_length)
_ARC = [
    ("baseline", None, 0.85, 0.00, 1800),
    ("step24", 24, 0.02, 0.62, 40),
    ("step48", 48, 0.12, 0.15, 220),
    ("final", 64, 0.05, 0.04, 140),
]


def _sample_text(ok: bool, gold: str, collapse: bool, rng: random.Random) -> tuple[str, str | None, str, int]:
    """(text, extracted, failure_kind|null, gen_tokens)."""
    if ok:
        return (
            f"Let me work through this.\n\n...\n\nThe answer is {gold}.",
            gold, None, int(1800 * rng.uniform(0.7, 1.3)),
        )
    if collapse and rng.random() < 0.5:
        return "", None, "empty", rng.randint(1, 8)
    wrong = str(int(gold) + rng.randint(1, 9)) if gold.isdigit() else "?"
    if collapse and rng.random() < 0.15:
        return f"The answer is {wrong}." * 60, wrong, "truncation", 4096
    return f"The answer is {wrong}.", wrong, "wrong_answer", rng.randint(30, 400)


def _divergence_row(rng: random.Random, row_id: str, base_id: str, ckpt_id: str,
                    sample_idx: int, prompt_len: int, ramp_nats: float,
                    skip_base: float, skip_ckpt: float) -> dict:
    n = rng.randint(300, 500)
    lp_base = [rng.gauss(-0.05, 0.12) for _ in range(n)]
    onset = rng.randint(60, 180)
    # per-token delta: baseline noise, then a negative ramp once the collapse
    # bites - mirrors what real SFT collapse looked like in the nats trace
    delta = [
        rng.gauss(0.0, 0.06) if i < onset else rng.gauss(-ramp_nats, 0.4)
        for i in range(n)
    ]
    cum, pos, tau = 0.0, None, 2.0
    for i, d in enumerate(delta):
        cum += d
        if cum <= -tau:
            pos = i
            break
    wins = [sum(delta[i:i + 32]) / min(32, n - i) for i in range(n)]
    sum_nats = sum(delta)
    return {
        "base_run_id": base_id, "ckpt_run_id": ckpt_id, "row_id": row_id,
        "sample_idx": sample_idx, "prompt_len": prompt_len, "n": n,
        "lp_base": lp_base,
        "lp_ckpt": [a + b for a, b in zip(lp_base, delta)],
        "delta": delta, "sum_nats": sum_nats, "mean_nats": sum_nats / n,
        "divergence_pos": pos,
        "win_argmin": min(range(n), key=lambda i: wins[i]),
        "p_skip_base": skip_base, "p_skip_ckpt": skip_ckpt,
        "cost_usd": n * 2 * 2e-6,
    }


def seed_synthetic(store: Store, *, seed: int = 0) -> str:
    """Write the synthetic manifest, four-run collapse arc, samples, divergence
    traces, and a falling-loss series. Returns the study name. Idempotent-ish:
    reruns overwrite the same run ids (labels clash -> warnings suppressed by
    deterministic ids)."""
    rng = random.Random(seed)
    rows = [
        {
            "row_id": f"synthetic-{i}",
            "benchmark": "synthetic",
            "messages": [{"role": "user", "content": q}],
            "gold": g, "grader_id": "exact",
            "source_ids": {"benchmark": "synthetic"},
        }
        for i, (q, g) in enumerate(_QUESTIONS)
    ]
    from flipbook.manifest import manifest_hash
    man = Manifest(
        name=MANIFEST_NAME, manifest_hash=manifest_hash(rows),
        spec={"benchmarks": {"synthetic": {"n": len(rows), "source": "generated"}},
              "seed": seed, "synthetic": True},
        created_at=datetime.now(UTC).isoformat(), rows=rows,
    )
    store.put_manifest(man.to_doc(), man.rows)

    run_ids: dict[str, str] = {}
    for label, step, acc, p_skip, gen_len in _ARC:
        rid = f"syn_{label}".ljust(12, "0")
        run_ids[label] = rid
        samples = []
        for r in rows:
            # per-row skill around the step's mean -> natural pass@2 > pass@1
            # spread and a mix of hard regressions vs near-misses
            skill = min(0.99, max(0.0, rng.gauss(acc, 0.10)))
            correct = sum(rng.random() < skill for _ in range(2))
            for sidx in range(2):
                ok = sidx < correct
                text, ext, fk, gtok = _sample_text(
                    ok, r["gold"], collapse=step is not None, rng=rng)
                samples.append({
                    "run_id": rid, "row_id": r["row_id"], "sample_idx": sidx,
                    "text": text, "prompt_tokens": rng.randint(80, 140),
                    "gen_tokens": gtok,
                    "stop_reason": "max_tokens" if fk == "truncation" else "stop",
                    "verdict": 1.0 if ok else 0.0, "extracted": ext,
                    "failure_kind": fk, "grade_note": None, "error": None,
                    "est_cost_usd": gtok * 2e-6,
                    "token_ids": list(range(gtok)), "token_logprobs": None,
                })
        store.put_samples(rid, samples)
        step_path = f"sampler_weights/{step:06d}" if step else "sampler_weights/final"
        n_ok = sum(s["verdict"] for s in samples)
        store.put_run({
            "run_id": rid, "config_fp": f"synthetic-{label}",
            "created_at": datetime.now(UTC).isoformat(),
            "manifest_hash": man.manifest_hash, "study": STUDY, "source": "synthetic",
            "label": label, "model_id": "synthetic/Inkling-Small",
            "checkpoint_path": None if step is None else f"tinker://synthetic/{step_path}",
            "base_model": "synthetic/Inkling-Small" if step else None,
            "train_step": step,
            "renderer": "tml_v0", "renderer_version": "synthetic",
            "effort": 0.9, "temperature": 0.6, "max_tokens": 4096,
            "k": 2, "seed": seed,
            "grader_id": "benchmark-native", "grader_version": "synthetic",
            "sdk_versions": {}, "state": "done",
            "provenance": {
                "evaluated_cells": len(samples),
                "train_step_measured": step,
                "n_samples": len(samples),
                "pass1": n_ok / len(samples),
                "est_cost_usd": sum(s["est_cost_usd"] for s in samples),
            },
        })

    # divergence traces: the -1800-nat cliff at step24, a milder one at final
    for label, ramp, skip_c in (("step24", 4.5, 0.62), ("final", 0.9, 0.04)):
        store.put_divergence(
            run_ids["baseline"], run_ids[label],
            [_divergence_row(rng, r["row_id"], run_ids["baseline"], run_ids[label],
                             0, 110, ramp, 0.001, skip_c) for r in rows],
        )

    # train loss keeps falling while eval craters - the story's whole point
    store.put_training_metrics(STUDY, [
        {"study": STUDY, "step": s, "key": "train/nll",
         "value": 2.1 * math.exp(-s / 22.0) + 0.25}
        for s in range(0, 65, 4)
    ])
    return STUDY
