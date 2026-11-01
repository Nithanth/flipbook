import pytest

from flipbook.config import RunConfig, run_id
from flipbook.manifest import Manifest, manifest_hash
from flipbook.runner import LintFailed, _failure_kind, _prior_gen_mean, _train_step, evaluate
from flipbook.store import Store


def _store_with_manifest(tmp_path, n=4):
    store = Store(tmp_path)
    rows = [
        {
            "row_id": f"gsm8k:{i:012d}", "benchmark": "gsm8k",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "gold": "1", "grader_id": "gsm8k", "source_ids": {},
        }
        for i in range(n)
    ]
    m = Manifest(name="m", manifest_hash=manifest_hash(rows), spec={},
                 created_at="t", rows=rows)
    store.put_manifest(m.to_doc(), m.rows)
    return store, m


def _cfg(mhash, **kw):
    return RunConfig(
        model="thinkingmachines/Inkling-Small", manifest_hash=mhash,
        effort=0.9, temperature=0.0, max_tokens=32768, k=2, seed=0,
        renderer="tml_v0", **kw,
    )


def test_failure_kind_ordering():
    assert _failure_kind(1.0, "x", "length") is None
    assert _failure_kind(0.0, "x", "length") == "truncation"
    assert _failure_kind(0.0, None, "stop") == "parse"
    assert _failure_kind(0.0, "x", "stop") == "wrong_answer"


def test_train_step_from_path():
    assert _train_step("tinker://r/sampler_weights/000024") == 24
    assert _train_step("tinker://r/sampler_weights/final") is None
    assert _train_step(None) is None


def test_prior_gen_mean_nearest_effort(tmp_path):
    store, m = _store_with_manifest(tmp_path)
    store.put_run({"run_id": "low", "manifest_hash": m.manifest_hash, "effort": 0.2})
    store.put_run({"run_id": "high", "manifest_hash": m.manifest_hash, "effort": 0.9})
    store.put_samples("low", [
        {"run_id": "low", "row_id": f"gsm8k:{i:012d}", "sample_idx": 0, "gen_tokens": 100}
        for i in range(4)
    ])
    store.put_samples("high", [
        {"run_id": "high", "row_id": f"gsm8k:{i:012d}", "sample_idx": 0, "gen_tokens": 500}
        for i in range(4)
    ])
    mean, ids = _prior_gen_mean(store, m.manifest_hash, effort=0.8)
    assert mean == 500
    assert ids == ["high"]
    assert _prior_gen_mean(store, "other_hash", effort=0.9) is None


def test_forecast_counts_missing_cells(tmp_path):
    store, m = _store_with_manifest(tmp_path, n=4)
    cfg = _cfg(m.manifest_hash)
    # one sample already stored -> cells = 4*2 - 1 = 7
    store.put_run({"run_id": run_id(cfg), "manifest_hash": m.manifest_hash, "effort": 0.9})
    store.put_samples(run_id(cfg), [
        {"run_id": run_id(cfg), "row_id": "gsm8k:000000000000",
         "sample_idx": 0, "gen_tokens": 10}
    ])
    s = evaluate(cfg, store, forecast=True)
    assert s.forecast.cells == 7
    assert s.forecast.prior == "store"
    assert s.forecast.usd_discount is not None
    assert s.pass1 is None


def test_lint_aborts_before_sampling(tmp_path):
    from dataclasses import replace

    store, m = _store_with_manifest(tmp_path)
    cfg = replace(_cfg(m.manifest_hash), effort=None)  # Inkling without effort -> error
    with pytest.raises(LintFailed):
        evaluate(cfg, store)
