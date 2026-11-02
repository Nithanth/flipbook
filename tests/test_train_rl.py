import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load():
    path = Path(__file__).parent.parent / "scripts" / "train_rl.py"
    spec = importlib.util.spec_from_file_location("train_rl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_final_checkpoint_gets_evaluated(tmp_path, monkeypatch):
    pytest.importorskip("tinker_cookbook")
    trl = _load()
    args = SimpleNamespace(
        env="arithmetic", log_dir=tmp_path / "rl", model="thinkingmachines/Inkling-Small", manifest="mh",
        baseline=None, store=str(tmp_path / "store"), study="rl_smoke",
        steps=10, groups_per_batch=8, group_size=4, max_tokens=4096,
        temperature=1.0, lr=1e-5, lora_rank=32, kl_coef=0.0, seed=0,
        save_every=5, eval_every=5, k=2, diverge=False, effort_pair=False,
        yes=True,
    )
    steps_seen = []

    class FakeEval:
        def __init__(self, **kw):
            pass

        async def __call__(self, client, step=None):
            steps_seen.append(step)
            return {}

    monkeypatch.setattr(trl, "RegressionEvaluator", FakeEval)
    monkeypatch.setattr(trl, "build_dataset", lambda *a: object())
    monkeypatch.setattr(trl, "Config", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(trl, "rl_main", lambda config: asyncio.sleep(0))
    monkeypatch.setattr(trl.cli_utils, "check_log_dir", lambda *a, **k: None)
    monkeypatch.setattr(
        trl.checkpoint_utils, "get_last_checkpoint",
        lambda *a, **k: SimpleNamespace(
            sampler_path="tinker://x/sampler_weights/final"),
    )
    monkeypatch.setattr(trl, "import_metrics", lambda *a, **k: 0)

    class SC:
        def create_sampling_client(self, **kw):
            return object()

    monkeypatch.setattr("tinker.ServiceClient", SC)
    assert trl.run_train(args) == 0
    # the RL loop never evals sampler_weights/final; the recipe must, at the
    # last step, not at calls*eval_every
    assert steps_seen == [10]
