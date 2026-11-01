from dataclasses import replace

from flipbook.config import RunConfig, config_fp, run_id


def _cfg(**kw):
    return RunConfig(
        model="thinkingmachines/Inkling-Small",
        manifest_hash="abc123",
        effort=0.9,
        temperature=0.6,
        max_tokens=512,
        k=4,
        seed=0,
        renderer="tml_v0",
        **kw,
    )


def test_fp_ignores_label():
    assert config_fp(_cfg(label="a")) == config_fp(_cfg(label="b")) == config_fp(_cfg())


def test_fp_tracks_study():
    assert config_fp(_cfg(study="control")) != config_fp(_cfg(study="treatment"))


def test_fp_tracks_eval_params():
    assert config_fp(replace(_cfg(), temperature=0.7)) != config_fp(_cfg())


def test_run_id_is_fp_prefix():
    cfg = _cfg()
    assert run_id(cfg) == config_fp(cfg)[:12]
    assert len(run_id(cfg)) == 12


def test_same_config_same_fp():
    assert config_fp(_cfg(study="s", label="l")) == config_fp(_cfg(study="s", label="l"))


def test_replace_roundtrip():
    cfg = _cfg()
    assert config_fp(replace(cfg)) == config_fp(cfg)
