from flipbook.cli import _report, main
from flipbook.store import Store


def _study(store, base_vs, ckpt_vs):
    store.put_run({"run_id": "b", "study": "s", "manifest_hash": "mh",
                   "label": "baseline", "k": 1})
    store.put_run({"run_id": "c", "study": "s", "manifest_hash": "mh",
                   "label": "step5", "train_step": 5, "k": 1})
    for rid, vs in (("b", base_vs), ("c", ckpt_vs)):
        store.put_samples(rid, [
            {
                "run_id": rid, "row_id": f"r{i}", "sample_idx": 0,
                "text": "t", "prompt_tokens": 1, "gen_tokens": 10,
                "stop_reason": "stop", "verdict": float(v),
                "extracted": None, "failure_kind": None if v else "wrong_answer",
                "error": None, "est_cost_usd": 0.0,
            }
            for i, v in enumerate(vs)
        ])


def test_report_ci_spanning_zero_is_no_change(tmp_path, capsys):
    # 0.95 -> 0.90 on 20 paired rows: the point drop is 0.05 but the
    # bootstrap CI still spans zero, so the verdict must not call it a regression
    store = Store(tmp_path)
    _study(store, [1] * 20, [1] * 19 + [0])
    _report(store, "s")
    out = capsys.readouterr().out
    assert "no eval-visible change" in out
    assert "includes zero" in out


def test_report_ci_excluding_zero_is_regressed(tmp_path, capsys):
    store = Store(tmp_path)
    _study(store, [1] * 20, [0] * 20)
    _report(store, "s")
    out = capsys.readouterr().out
    assert "the fine-tune regressed this eval" in out


def _reward_metrics(store, first, last, n=10):
    import numpy as np

    vals = np.linspace(first, last, n).tolist()
    store.put_training_metrics("s", [
        {"step": i, "key": "env/all/reward/total", "value": v}
        for i, v in enumerate(vals)
    ])


def test_report_reward_saturated(tmp_path, capsys):
    store = Store(tmp_path)
    _study(store, [1] * 20, [1] * 19 + [0])
    _reward_metrics(store, 1.0, 1.0)
    _report(store, "s")
    out = capsys.readouterr().out
    assert "reward: 1.00 -> 1.00" in out
    assert "saturated" in out


def test_report_reward_up_eval_down(tmp_path, capsys):
    store = Store(tmp_path)
    _study(store, [1] * 20, [0] * 20)
    _reward_metrics(store, 0.2, 0.8)
    _report(store, "s")
    out = capsys.readouterr().out
    assert "reward up, held-out eval down" in out
    assert "fitting the reward" in out



def test_report_skips_runs_still_sampling(tmp_path, capsys):
    # mid-training report: the newest checkpoint is still writing samples and
    # must not narrate as a data point
    store = Store(tmp_path)
    _study(store, [1] * 20, [1] * 19 + [0])
    store.put_run({"run_id": "c2", "study": "s", "manifest_hash": "mh",
                   "label": "step10", "train_step": 10, "k": 1,
                   "state": "running"})
    store.put_samples("c2", [
        {"run_id": "c2", "row_id": f"r{i}", "sample_idx": 0,
         "text": "t", "prompt_tokens": 1, "gen_tokens": 10,
         "stop_reason": "stop", "verdict": 0.0,
         "extracted": None, "failure_kind": "wrong_answer",
         "error": None, "est_cost_usd": 0.0}
        for i in range(20)
    ])
    _report(store, "s")
    out = capsys.readouterr().out
    assert "step 10" not in out
    assert "step 5" in out


def test_report_json_is_the_same_read(tmp_path, capsys):
    import json
    store = Store(tmp_path)
    _study(store, [1] * 20, [0] * 20)
    _report(store, "s")
    text = capsys.readouterr().out
    _report(store, "s", as_json=True)
    d = json.loads(capsys.readouterr().out)
    assert d["verdict"] in text
    assert all(b in text for b in d["bullets"])
    assert [p["step"] for p in d["pass1"]] == [0, 5]
    assert d["delta_ci"][1] < 0 and d["compare"].startswith("flipbook compare s/")


def _status_study(store):
    # baseline + step0 (the base weights again) + step5, nothing scored yet
    store.put_run({"run_id": "b", "study": "s", "manifest_hash": "mh",
                   "label": "baseline", "k": 1})
    store.put_run({"run_id": "c0", "study": "s", "manifest_hash": "mh",
                   "label": "step0", "train_step": 0, "k": 1,
                   "provenance": {"train_step_measured": 0}})
    store.put_run({"run_id": "c5", "study": "s", "manifest_hash": "mh",
                   "label": "step5", "train_step": 5, "k": 1,
                   "provenance": {"train_step_measured": 5}})


def test_status_effort_hint_uses_canonical_pair(tmp_path, capsys):
    store = Store(tmp_path)
    _status_study(store)
    main(["status", "--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert "--pair 0.2,0.9" in out
    assert "LOW,HIGH" not in out


def test_status_step0_is_not_missing_divergence(tmp_path, capsys):
    # step-0 weights ARE the baseline, so divergence has nothing to score
    store = Store(tmp_path)
    _status_study(store)
    main(["status", "--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert "--ckpt s/step5" in out
    assert "--ckpt s/step0" not in out


def test_status_counts_train_step_and_step0_measured(tmp_path, capsys):
    # step0 carries only the in-loop measured step; step5 only the
    # ref-resolved train_step - both count as checkpoints
    store = Store(tmp_path)
    store.put_run({"run_id": "b", "study": "s", "manifest_hash": "mh",
                   "label": "baseline", "k": 1})
    store.put_run({"run_id": "c0", "study": "s", "manifest_hash": "mh",
                   "label": "step0", "k": 1,
                   "provenance": {"train_step_measured": 0}})
    store.put_run({"run_id": "c5", "study": "s", "manifest_hash": "mh",
                   "label": "step5", "train_step": 5, "k": 1})
    main(["status", "--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert "checkpoints  2 (steps 0..5)" in out
    assert "missing: step 0, 5" in out  # effort gap lists both
    assert "missing: step 5" in out    # divergence skips the step-0 weights


def test_status_warns_on_duplicate_labels(tmp_path, capsys):
    # two 'baseline' runs make the study/baseline ref ambiguous
    store = Store(tmp_path)
    _status_study(store)
    store.put_run({"run_id": "b2", "study": "s", "manifest_hash": "mh",
                   "label": "baseline", "k": 1})
    main(["status", "--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert "warning: duplicate label 'baseline' - use run_id refs" in out
