import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load():
    path = Path(__file__).parent.parent / "scripts" / "train_rl.py"
    spec = importlib.util.spec_from_file_location("train_rl", path)
    mod = importlib.util.module_from_spec(spec)
    # chz resolves the defining module via sys.modules[cls.__module__]
    sys.modules["train_rl"] = mod
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
        save_every=5, eval_every=5, k=2, diverge=False, effort_pair=False, length_coef=0.0,
        short_tokens=0, test_size=0, resume=False, yes=True,
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


def test_length_penalized_env_step():
    trl = _load()
    from tinker_cookbook.recipes.math_rl import math_env
    from tinker_cookbook.rl.types import StepResult

    async def fake_step(self, action, *, extra=None):
        return StepResult(
            reward=1.0, episode_done=True, next_observation=None,
            next_stop_condition=None, metrics={}, logs={},
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(math_env.ProblemEnv, "step", fake_step)
        env = trl.LengthPenalizedEnv(
            "q", "1", renderer=SimpleNamespace(), length_coef=0.5, max_tokens=100
        )
        r = asyncio.run(env.step(list(range(50))))
    assert r.reward == 0.75
    assert r.metrics["length_penalty"] == 0.25


def test_build_dataset_length_penalized():
    trl = _load()
    args = SimpleNamespace(
        env="gsm8k", model="m", steps=1, groups_per_batch=2, group_size=2,
        seed=0, length_coef=0.5, short_tokens=0, max_tokens=100, test_size=8,
    )
    b = trl.build_dataset(args, "tml_v0")
    assert isinstance(b, trl.ShapedGsm8kBuilder)
    assert b.length_coef == 0.5 and b.test_size == 8
    # gsm8k keeps this builder at coef 0 so the test-split cap still applies
    args.length_coef = 0.0
    assert trl.build_dataset(args, "tml_v0").test_size == 8
    args.env = "math"
    assert not isinstance(trl.build_dataset(args, "tml_v0"), trl.ShapedGsm8kBuilder)


def test_length_coef_rejects_non_gsm8k(monkeypatch):
    trl = _load()
    monkeypatch.setattr(
        "sys.argv",
        ["train_rl.py", "--env", "arithmetic", "--manifest", "m", "--length-coef", "0.5"],
    )
    with pytest.raises(SystemExit):
        trl.main()



def _short_env(trl, **kw):
    return trl.ShortAnswerEnv(
        "q", "1", renderer=SimpleNamespace(), short_tokens=40, **kw
    )


@pytest.fixture
def stub_parent(monkeypatch):
    from tinker_cookbook.recipes.math_rl import math_env
    from tinker_cookbook.rl.types import StepResult

    state = {"reward": 1.0, "metrics": {"format": 1.0, "correct": 1.0}}

    async def fake_step(self, action, *, extra=None):
        return StepResult(
            reward=state["reward"], episode_done=True, next_observation=None,
            next_stop_condition=None, metrics=dict(state["metrics"]), logs={},
        )

    monkeypatch.setattr(math_env.ProblemEnv, "step", fake_step)
    return state


def test_short_answer_env_short(stub_parent):
    trl = _load()
    r = asyncio.run(_short_env(trl).step(list(range(30))))
    assert r.reward == 1.0 and r.metrics["short"] == 1.0


def test_short_answer_env_long(stub_parent):
    trl = _load()
    r = asyncio.run(_short_env(trl).step(list(range(50))))
    assert r.reward == 0.0 and r.metrics["short"] == 0.0


def test_short_answer_env_malformed(stub_parent):
    trl = _load()
    stub_parent["metrics"] = {"format": 0.0, "correct": 0.0}
    r = asyncio.run(_short_env(trl).step(list(range(30))))
    assert r.reward == 0.0


def test_build_dataset_short_tokens():
    trl = _load()
    args = SimpleNamespace(
        env="gsm8k", model="m", steps=1, groups_per_batch=2, group_size=2,
        seed=0, length_coef=0.0, short_tokens=256, max_tokens=100, test_size=8,
    )
    b = trl.build_dataset(args, "tml_v0")
    assert isinstance(b, trl.ShapedGsm8kBuilder)
    assert b.short_tokens == 256


def test_reward_shapes_mutually_exclusive(monkeypatch):
    trl = _load()
    monkeypatch.setattr(
        "sys.argv",
        ["train_rl.py", "--env", "gsm8k", "--manifest", "m",
         "--length-coef", "0.5", "--short-tokens", "256"],
    )
    with pytest.raises(SystemExit, match="alternative reward shapes"):
        trl.main()



def test_start_batch_and_resume_guard(tmp_path):
    trl = _load()
    args = SimpleNamespace(resume=False, log_dir=tmp_path, steps=40)
    assert trl._start_batch(args) == 0
    args.resume = True
    assert trl._start_batch(args) == 0
    import json

    (tmp_path / "checkpoints.jsonl").write_text(
        json.dumps({"name": "000020", "batch": 20,
                    "state_path": "tinker://x/weights/000020",
                    "sampler_path": "tinker://x/sampler_weights/000020"}) + "\n")
    assert trl._start_batch(args) == 20
    args.steps = 20
    with pytest.raises(SystemExit, match="nothing to do"):
        trl.build_config(args)
