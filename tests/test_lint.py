from dataclasses import replace

from flipbook.config import RunConfig
from flipbook.lint import lint


def _cfg(**kw):
    return RunConfig(
        model="thinkingmachines/Inkling-Small",
        manifest_hash="h",
        effort=0.9,
        temperature=0.7,
        max_tokens=32768,
        k=4,
        seed=0,
        renderer="tml_v0",
        **kw,
    )


def _codes(findings):
    return {f.code for f in findings}


def test_clean_config():
    assert not any(f.level == "error" for f in lint(_cfg()))


def test_inkling_needs_tml_v0():
    fs = lint(replace(_cfg(), renderer="qwen3"))
    assert "E_INKLING_RENDERER" in _codes(fs)
    assert "E_EFFORT_NON_TML" in _codes(fs)


def test_inkling_needs_effort():
    assert "E_INKLING_NO_EFFORT" in _codes(lint(replace(_cfg(), effort=None)))


def test_effort_range():
    assert "E_EFFORT_RANGE" in _codes(lint(replace(_cfg(), effort=1.0)))


def test_effort_on_non_tml_is_error():
    cfg = replace(_cfg(), model="Qwen/Qwen3-8B", renderer="qwen3")
    assert "E_EFFORT_NON_TML" in _codes(lint(cfg))


def test_budget_warn():
    assert "W_BUDGET" in _codes(lint(replace(_cfg(), max_tokens=4096)))


def test_unknown_model_prices_warn():
    cfg = replace(_cfg(), model="Qwen/Qwen3-8B", renderer="qwen3", effort=None)
    fs = lint(cfg)
    assert "W_NO_PRICES" in _codes(fs)


def test_unresolvable_grader_ids_are_errors():
    fs = lint(_cfg(), grader_ids=["gsm8k", "nope", "regex:("])
    assert "E_UNKNOWN_GRADER" in _codes(fs)
    assert not any(f.level == "error" for f in lint(_cfg(), grader_ids=["gsm8k", "regex:\\d+"]))


def test_unresolvable_tinker_skips_inkling_checks():
    fs = lint(replace(_cfg(), model="tinker://x/sampler_weights/000024"))
    assert "E_INKLING_RENDERER" not in _codes(fs)
    assert "E_INKLING_NO_EFFORT" not in _codes(fs)


def test_explicit_base_model_restores_checks():
    cfg = replace(_cfg(), model="tinker://x/sampler_weights/000024", effort=None)
    assert "E_INKLING_NO_EFFORT" in _codes(lint(cfg, base_model="thinkingmachines/Inkling"))
