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
    def __init__(self, tokens, model_path=MODEL):
        self.tokens = tokens
        self.model_path = model_path
    async def sample_async(self, prompt, num_samples, sampling_params, **kw):
        seq = SimpleNamespace(tokens=self.tokens, stop_reason="stop", logprobs=None)
        return SimpleNamespace(sequences=[seq] * num_samples)
    async def _get_sampler_submit(self):
        return SimpleNamespace(model_path=self.model_path)
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
    # explicit step so the fake's base-model path isn't short-circuited as step 0
    metrics = asyncio.run(ev(fake, step=8))
    assert metrics["flipbook/m/pass1"] == 1.0
    assert metrics["flipbook/m/delta_vs_base"] == 0.0
    assert "flipbook/m/delta_ci_lo" in metrics
    assert "flipbook/m/truncation_rate" in metrics
    assert "flipbook/m/cost_usd" not in metrics or metrics["flipbook/m/cost_usd"] >= 0
    assert ev.calls == 1
    # a ckpt run record exists, with the measured step patched in
    import json
    runs = [json.loads(f.read_text()) for f in (store.path / "runs").glob("*.json")]
    ckpt = next(r for r in runs if r["provenance"].get("train_step_measured") == 8)
    assert ckpt["provenance"]["baseline_run_id"] == base_rid
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


def test_step0_eval_is_the_baseline(tmp_path, monkeypatch):
    # step-0 weights are the untouched base model: billing that eval as the
    # baseline keeps _ensure_baseline from sampling the same model twice
    store, _m = _store_with_manifest(tmp_path)
    fake = FakeSampler(_cont_tokens())
    import json

    import flipbook.evaluator as evmod
    real = evmod.evaluate_async
    seen = []

    async def with_fake(cfg, store, **kw):
        seen.append(cfg.label)
        kw.setdefault("sampling_client", fake)
        return await real(cfg, store, **kw)

    monkeypatch.setattr(evmod, "evaluate_async", with_fake)
    ev = RegressionEvaluator(
        manifest="m", baseline=MODEL, store=store, eval_every=8,
        k=2, temperature=0.0, max_tokens=256,
        diverge=False, effort_pair=None,
    )
    metrics = asyncio.run(ev(fake))
    assert seen == ["baseline"]  # exactly one eval, under the baseline label
    runs = [json.loads(f.read_text()) for f in (store.path / "runs").glob("*.json")]
    assert len(runs) == 1
    assert runs[0]["label"] == "baseline"
    assert runs[0].get("train_step") is None  # API/GUI identify baseline this way
    assert metrics["flipbook/m/pass1"] == 1.0
    assert metrics["flipbook/m/delta_vs_base"] == 0.0


def test_step0_reuses_a_run_ref_baseline(tmp_path, monkeypatch):
    # baseline given as a run ref of the same base model: step 0 samples
    # nothing and charts the baseline's own numbers
    store, m = _store_with_manifest(tmp_path)
    fake = FakeSampler(_cont_tokens())
    base_cfg = _cfg(m.manifest_hash)
    asyncio.run(evaluate_async(base_cfg, store, sampling_client=fake))
    import flipbook.evaluator as evmod
    seen = []

    async def spy(cfg, store, **kw):
        seen.append(cfg.label)
        raise AssertionError("step 0 must not sample")

    monkeypatch.setattr(evmod, "evaluate_async", spy)
    ev = RegressionEvaluator(
        manifest="m", baseline=run_id(base_cfg), store=store, eval_every=8,
        k=2, temperature=0.0, max_tokens=256, diverge=True, effort_pair=None, study="s",
    )
    metrics = asyncio.run(ev(fake))
    assert seen == []
    assert metrics["flipbook/m/pass1"] == 1.0
    assert "flipbook/m/divergence_mean_nats" not in metrics
    assert store.training_metrics("s").num_rows > 0



def test_already_evaluated_step_is_reused(tmp_path, monkeypatch):
    # a resumed run asks for a step the store already has: recompute metrics
    # from the stored run, never sample it again
    store, m = _store_with_manifest(tmp_path)
    fake = FakeSampler(_cont_tokens())
    base_cfg = _cfg(m.manifest_hash)
    asyncio.run(evaluate_async(base_cfg, store, sampling_client=fake))
    base_rid = run_id(base_cfg)
    kw = {"manifest": "m", "baseline": base_rid, "store": store, "eval_every": 8,
          "efforts": (0.9,), "k": 2, "temperature": 0.0, "max_tokens": 256,
          "diverge": False, "effort_pair": None, "study": "s"}
    asyncio.run(RegressionEvaluator(**kw)(fake, step=8))
    import flipbook.evaluator as evmod

    async def spy(cfg, store, **kw):
        raise AssertionError("already-evaluated step must not sample")

    monkeypatch.setattr(evmod, "evaluate_async", spy)
    metrics = asyncio.run(RegressionEvaluator(**kw)(fake, step=8))
    assert metrics["flipbook/m/pass1"] == 1.0


def test_start_step_offsets_calls(tmp_path):
    store, _m = _store_with_manifest(tmp_path)
    ev = RegressionEvaluator(manifest="m", baseline="x", store=store,
                             eval_every=8, start_step=16)
    assert ev.calls == 2
    with pytest.raises(ValueError):
        RegressionEvaluator(manifest="m", baseline="x", store=store,
                            eval_every=8, start_step=5)



def test_ephemeral_path_runs_under_the_canonical_checkpoint(tmp_path):
    # in-loop samplers report sampler_weights/ephemeral_N, which expires; the
    # run must be billed to the numbered checkpoint so `flipbook eval` dedupes
    store, m = _store_with_manifest(tmp_path)
    eph = "tinker://x:train:0/sampler_weights/ephemeral_3"
    canon = "tinker://x:train:0/sampler_weights/000008"
    fake = FakeSampler(_cont_tokens(), model_path=eph)
    base_cfg = _cfg(m.manifest_hash)
    asyncio.run(evaluate_async(base_cfg, store, sampling_client=fake))
    base_rid = run_id(base_cfg)
    ev = RegressionEvaluator(
        manifest="m", baseline=base_rid, store=store, eval_every=8,
        efforts=(0.9,), k=2, temperature=0.0, max_tokens=256,
        diverge=False, effort_pair=None)
    # resolve_model on a tinker:// path needs REST; the evaluator resolves
    # once, so pre-set it as it would look after a real first call
    from flipbook.runner import Resolved
    ev._resolved = Resolved(base_model=MODEL, renderer="tml_v0", checkpoint_path=eph)
    asyncio.run(ev(fake, step=8))
    import json
    canon_cfg = RunConfig(
        model=canon, manifest_hash=m.manifest_hash, effort=0.9,
        temperature=0.0, max_tokens=256, k=2, seed=0, renderer="tml_v0",
        label="step8")
    want = run_id(canon_cfg)
    rec = json.loads((store.path / "runs" / f"{want}.json").read_text())
    assert rec["checkpoint_path"] == canon
    assert rec["provenance"]["baseline_run_id"] == base_rid
