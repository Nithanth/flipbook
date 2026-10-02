import asyncio
from types import SimpleNamespace

import pytest

from flipbook.config import RunConfig, run_id
from flipbook.evaluator import RegressionEvaluator
from flipbook.manifest import Manifest, manifest_hash
from flipbook.runner import evaluate_async
from flipbook.store import Store

MODEL = "thinkingmachines/Inkling-Small"


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
    m = Manifest(name="m", manifest_hash=manifest_hash(rows), spec={}, created_at="t", rows=rows)
    store.put_manifest(m.to_doc(), m.rows)
    return store, m


def _cont_tokens():
    """Real tml_v0 assistant turn so parse_response produces clean text."""
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    r = get_renderer("tml_v0", get_tokenizer(MODEL))
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "\\boxed{1}"}]
    mi, _ = r.build_supervised_example(msgs, effort=0.9)
    toks = mi.to_ints()
    return toks[len(r.build_generation_prompt(msgs[:1], effort=0.9).to_ints()):]


class FakeSampler:
    """Answers every prompt with the same canned sequence."""
    def __init__(self, tokens):
        self.tokens = tokens
    async def sample_async(self, prompt, num_samples, sampling_params, **kw):
        seq = SimpleNamespace(tokens=self.tokens, stop_reason="stop", logprobs=None)
        return SimpleNamespace(sequences=[seq] * num_samples)
    async def _get_sampler_submit(self):
        return SimpleNamespace(model_path=MODEL)
    async def compute_logprobs_async(self, seq):
        return [None] + [-0.5] * (len(seq.to_ints()) - 1)


def _cfg(mhash):
    return RunConfig(
        model=MODEL, manifest_hash=mhash, effort=0.9, temperature=0.0,
        max_tokens=256, k=2, seed=0, renderer="tml_v0",
    )


def test_evaluator_writes_runs_and_metrics(tmp_path):
    pytest.importorskip("tinker_cookbook")
    store, m = _store_with_manifest(tmp_path)
    fake = FakeSampler(_cont_tokens())
    # pre-seed the baseline so the evaluator reuses it, never re-samples
    base_cfg = _cfg(m.manifest_hash)
    asyncio.run(evaluate_async(base_cfg, store, sampling_client=fake))
    base_rid = run_id(base_cfg)
    ev = RegressionEvaluator(
        manifest="m", baseline=base_rid, store=store, eval_every=8,
        efforts=(0.9,), k=2, temperature=0.0, max_tokens=256,
        diverge=False, effort_pair=None,
    )
    metrics = asyncio.run(ev(fake))
    assert metrics["flipbook/m/pass1"] == 1.0
    assert metrics["flipbook/m/delta_vs_base"] == 0.0
    assert "flipbook/m/delta_ci_lo" in metrics
    assert "flipbook/m/truncation_rate" in metrics
    assert "flipbook/m/cost_usd" not in metrics or metrics["flipbook/m/cost_usd"] >= 0
    assert ev.calls == 1
    # a ckpt run record exists, with the measured step patched in
    import json
    runs = [json.loads(f.read_text()) for f in (store.path / "runs").glob("*.json")]
    assert any(r["provenance"].get("train_step_measured") == 0 for r in runs)
    # error path: missing manifest -> error metric, no raise
    ev2 = RegressionEvaluator(manifest="nope", baseline=base_rid, store=store,
                              diverge=False, effort_pair=None)
    m2 = asyncio.run(ev2(fake))
    assert m2 == {"flipbook/nope/error": 1.0}


def test_evaluator_baseline_via_model_name(tmp_path, monkeypatch):
    store, _m = _store_with_manifest(tmp_path)
    fake = FakeSampler(_cont_tokens())
    # the baseline needs its own real client in production; in tests we
    # intercept evaluate_async and hand it the fake (it never gets the
    # ckpt's snapshot client)
    import flipbook.evaluator as evmod
    real = evmod.evaluate_async
    async def with_fake(cfg, store, **kw):
        kw.setdefault("sampling_client", fake)
        return await real(cfg, store, **kw)
    monkeypatch.setattr(evmod, "evaluate_async", with_fake)
    ev = RegressionEvaluator(
        manifest="m", baseline=MODEL, store=store, eval_every=8,
        k=2, temperature=0.0, max_tokens=256,
        diverge=False, effort_pair=None,
    )
    metrics = asyncio.run(ev(fake))
    # baseline got evaluated once through the fake and memoized; the fake's
    # model_path IS the base name, so baseline and "ckpt" fingerprint to the
    # same run — correct dedupe, one record
    assert metrics["flipbook/m/pass1"] == 1.0
    assert metrics["flipbook/m/delta_vs_base"] == 0.0
    assert ev._baseline_rid is not None
    assert len(list((store.path / "runs").glob("*.json"))) == 1
