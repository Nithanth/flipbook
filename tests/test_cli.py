from flipbook.cli import _report
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
