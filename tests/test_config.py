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


def test_eval_config_file_merge(tmp_path):
    import argparse

    from flipbook.cli import _load_eval_file, _merge_eval_config

    f = tmp_path / "eval.toml"
    f.write_text('model = "thinkingmachines/Inkling-Small"\nmanifest = "aime30"\nk = 2\nlabel = "baseline"\n')
    assert _load_eval_file(str(f))["k"] == 2

    args = argparse.Namespace(config=str(f), model=None, manifest=None, k=4, label=None)
    _merge_eval_config(args, ["eval", "--config", str(f), "--k", "4"])
    assert args.model == "thinkingmachines/Inkling-Small"  # file supplies
    assert args.k == 4  # explicit flag wins over file's k=2
    assert args.label == "baseline"

    bad = tmp_path / "bad.toml"
    bad.write_text('model = "x"\nmanifest = "y"\nlskdjf = 1\n')
    try:
        _load_eval_file(str(bad))
        raise AssertionError("should have exited")
    except SystemExit as e:
        assert "lskdjf" in str(e)

    # a missing model/manifest is not a merge error anymore - the positional
    # ref can supply both, so validation moved into the eval dispatch
    args = argparse.Namespace(config=None, model=None, manifest=None)
    _merge_eval_config(args, ["eval"])
    assert args.model is None


def test_eval_ref_resolves_checkpoint_and_inherits(tmp_path):
    import argparse

    from flipbook.cli import _resolve_eval_ref
    from flipbook.store import Store

    store = Store(str(tmp_path))
    store.put_run({
        "run_id": "r1", "study": "s", "label": "step24",
        "checkpoint_path": "tinker://abc:train:0/sampler_weights/000024",
        "model_id": "thinkingmachines/Inkling-Small", "manifest_hash": "mh",
        "effort": 0.9, "temperature": 0.6, "k": 2, "seed": 0,
        "renderer": "tml_v0", "max_tokens": 8192,
    })
    base = {"ref": "s/step8", "model": None, "manifest": None, "effort": None,
            "temperature": 0.7, "max_tokens": 32768, "k": 4, "seed": 0,
            "renderer": None, "study": None, "label": None}
    args = argparse.Namespace(**base)
    _resolve_eval_ref(args, store, ["eval", "s/step8"])
    # the path is synthesized off the study's recorded sampler path, and the
    # rest of the config is inherited from that checkpoint's eval
    assert args.model == "tinker://abc:train:0/sampler_weights/000008"
    assert (args.manifest, args.k, args.temperature) == ("mh", 2, 0.6)
    assert (args.study, args.label) == ("s", "step8")

    args = argparse.Namespace(**{**base, "k": 1})
    _resolve_eval_ref(args, store, ["eval", "s/step8", "--k", "1"])
    assert args.k == 1  # explicit flag wins over the ref's k=2


def test_eval_ref_unknown_falls_back_to_model(tmp_path):
    import argparse

    from flipbook.cli import _resolve_eval_ref
    from flipbook.store import Store

    args = argparse.Namespace(
        ref="thinkingmachines/Inkling-Small", model=None, manifest=None,
        effort=None, temperature=0.7, max_tokens=32768, k=4, seed=0,
        renderer=None, study=None, label=None,
    )
    _resolve_eval_ref(args, Store(str(tmp_path)), ["eval"])
    assert args.model == "thinkingmachines/Inkling-Small"


def test_save_config_roundtrip(tmp_path):
    import argparse

    from flipbook.cli import _merge_eval_config, _write_eval_config

    out = tmp_path / "preset.toml"
    args = argparse.Namespace(
        model="thinkingmachines/Inkling-Small", manifest="aime30",
        effort=0.9, k=2, temperature=0.7, max_tokens=32768, seed=0,
        renderer=None, base_model=None, label="x", study=None, config=None,
    )
    _write_eval_config(args, str(out))
    text = out.read_text()
    assert 'k = 2' in text and 'renderer' not in text  # Nones are dropped

    args2 = argparse.Namespace(
        config=str(out), model=None, manifest=None, effort=None, k=None,
        temperature=None, max_tokens=None, seed=None, renderer=None,
        base_model=None, label=None, study=None,
    )
    _merge_eval_config(args2, ["eval", "--config", str(out)])
    assert args2.k == 2 and args2.label == "x" and args2.effort == 0.9
