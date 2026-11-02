import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load():
    path = Path(__file__).parents[1] / "scripts" / "train_demo.py"
    spec = importlib.util.spec_from_file_location("train_demo", path)
    mod = importlib.util.module_from_spec(spec)
    # chz resolves the defining module via sys.modules[cls.__module__]
    sys.modules["train_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


MODEL = "thinkingmachines/Inkling-Small"


def _cont_tokens():
    """Real tml_v0 assistant turn so parse_response produces clean text."""
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    r = get_renderer("tml_v0", get_tokenizer(MODEL))
    # a thinking segment so the parsed message exercises the list-content path
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "t"},
                {"type": "text", "text": "\\boxed{1}"}]}]
    mi, _ = r.build_supervised_example(msgs, effort=0.9)
    toks = mi.to_ints()
    return toks[len(r.build_generation_prompt(msgs[:1], effort=0.9).to_ints()):]


class FakeSampler:
    def __init__(self, tokens):
        self.tokens = tokens
    async def sample_async(self, prompt, num_samples, sampling_params, **kw):
        seq = SimpleNamespace(tokens=self.tokens, stop_reason="stop", logprobs=None)
        return SimpleNamespace(sequences=[seq] * num_samples)


FAKE_DS = [
    {"question": "q1", "answer": "work\n\n#### 1"},
    {"question": "q2", "answer": "work\n\n#### 2"},
]


def test_build_distilled_data(tmp_path, monkeypatch):
    td = _load()
    fake = FakeSampler(_cont_tokens())
    # data() can't hit HF in tests: the build reads rows[i]["question"|"answer"]
    monkeypatch.setattr(td, "load_benchmark_dataset", lambda *a, **k: FAKE_DS)
    out = tmp_path / "d.jsonl"
    rc = asyncio.run(td.build_distilled_data(
        out, n=2, seed=0, model=MODEL, max_tokens=256, pool=4, yes=True,
        client=fake))
    assert rc == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    # one prompt's gold is "1" (correct) and the other's is "2" (wrong answer)
    assert len(rows) == 1
    assistant = rows[0]["messages"][2]
    assert assistant["role"] == "assistant"
    assert isinstance(assistant["content"], list)


def test_build_distilled_data_dry_run(tmp_path, monkeypatch):
    td = _load()
    monkeypatch.setattr(td, "load_benchmark_dataset", lambda *a, **k: FAKE_DS)
    out = tmp_path / "d.jsonl"
    rc = asyncio.run(td.build_distilled_data(
        out, n=2, seed=0, model=MODEL, max_tokens=256, pool=4, yes=False,
        client=FakeSampler(_cont_tokens())))
    assert rc == 3
    assert not out.exists()



def test_build_config_holds_the_evaluator(tmp_path):
    td = _load()
    args = SimpleNamespace(
        model=MODEL, max_length=2048, batch_size=32,
        data=tmp_path / "d.jsonl", store=str(tmp_path / "store"),
        manifest="m", baseline=None, log_dir=tmp_path / "log",
        eval_every=8, k=2, diverge=False, effort_pair=None, study="s",
        lr=2e-4, lr_schedule="linear", epochs=2, lora_rank=32, max_steps=None,
        save_every=8,
    )
    _config, evaluator = td.build_config(args)
    assert isinstance(evaluator, td.RegressionEvaluator)
    assert evaluator.study == "s"
