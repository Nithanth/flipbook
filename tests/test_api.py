import json

import pytest
from fastapi.testclient import TestClient

from flipbook.api import create_app
from flipbook.store import Store


def _store(tmp_path) -> Store:
    store = Store(tmp_path)
    store.put_manifest(
        {"manifest_hash": "mh1", "name": "m", "version": "manifest/v2"},
        [
            {
                "row_id": f"r{i}", "benchmark": "b",
                "messages": [{"role": "user", "content": "q"}],
                "gold": "1", "grader_id": "g", "source_ids": {},
            }
            for i in range(4)
        ],
    )
    for rid, vs in (("a", [1, 1, 0, 1]), ("b", [0, 1, 0, 0])):
        store.put_run({"run_id": rid, "k": 1, "manifest_hash": "mh1"})
        store.put_samples(rid, [
            {
                "run_id": rid, "row_id": f"r{i}", "sample_idx": 0,
                "text": "t", "prompt_tokens": 10, "gen_tokens": 50,
                "stop_reason": "stop", "verdict": float(v),
                "extracted": "x" if v else None,
                "failure_kind": None if v else "wrong_answer",
                "error": None, "est_cost_usd": 0.001,
            }
            for i, v in enumerate(vs)
        ])
    store.put_divergence("a", "b", [
        {
            "row_id": "r0", "sample_idx": 0, "lp_base": [-0.1], "lp_ckpt": [-0.9],
            "delta": [-0.8], "sum_nats": -0.8, "mean_nats": -0.8,
            "divergence_pos": 0, "win_argmin": 0,
            "p_skip_base": 0.01, "p_skip_ckpt": 0.4, "cost_usd": 0.01,
        }
    ])
    store.put_effort("a", 0.2, 0.9, [
        {
            "run_id": "a", "row_id": "r0", "sample_idx": 0,
            "e_low": 0.2, "e_high": 0.9, "lp_low_sum": -50.0,
            "lp_high_sum": -30.0, "gap_nats": 20.0, "n": 100,
            "cost_usd": 0.01,
        }
    ])
    store.put_training_metrics("s1", [{"step": 8, "key": "loss", "value": 0.5}])
    return store


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_manifests_and_runs(tmp_path):
    _store(tmp_path)
    c = _client(tmp_path)
    ms = c.get("/api/manifests").json()
    assert ms[0]["manifest_hash"] == "mh1" and ms[0]["n_rows"] == 4
    runs = c.get("/api/runs").json()
    assert {r["run_id"] for r in runs} == {"a", "b"}


def test_samples_strips_heavy_cols(tmp_path):
    _store(tmp_path)
    c = _client(tmp_path)
    rows = c.get("/api/runs/a/samples").json()
    assert len(rows) == 4 and "verdict" in rows[0]
    assert "token_ids" not in rows[0] and "messages" not in rows[0]
    assert c.get("/api/runs/ghost/samples").status_code == 404


def test_compare_endpoint(tmp_path):
    _store(tmp_path)
    c = _client(tmp_path)
    p = c.get("/api/compare", params={"a": "a", "b": "b"}).json()
    assert p["n_pairs"] == 4 and p["acc_a"] == 0.75 and p["acc_b"] == 0.25
    assert p["agreement"]["a_only"] == 2
    assert p["comparability"]["ok"] and p["comparability"]["warnings"] == []


def test_compare_mismatched_manifest_is_422(tmp_path):
    store = _store(tmp_path)
    store.put_run({"run_id": "b", "k": 1, "manifest_hash": "mh2"})
    c = _client(tmp_path)
    r = c.get("/api/compare", params={"a": "a", "b": "b"})
    assert r.status_code == 422 and "manifests" in r.json()["detail"]


def test_compare_row(tmp_path):
    _store(tmp_path)
    c = _client(tmp_path)
    r = c.get("/api/compare/row", params={"a": "a", "b": "b", "row": "r1"})
    assert r.status_code == 200
    d = r.json()
    assert d["question"] == [{"role": "user", "content": "q"}]
    assert d["answer"] == "1"
    assert len(d["a"]) == 1 and len(d["b"]) == 1
    assert d["a"][0]["verdict"] == 1.0 and d["b"][0]["verdict"] == 1.0
    assert d["k_a"] == 1 and d["k_b"] == 1
    # fixture rows carry no token_ids, so there is no thinking to recover
    assert d["a"][0]["thinking"] is None
    assert d["a"][0]["text_clean"] == "t"
    assert c.get(
        "/api/compare/row", params={"a": "a", "b": "b", "row": "nope"}
    ).status_code == 404


def test_compare_row_thinking_and_text_clean(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.put_run({"run_id": "a", "k": 2, "manifest_hash": "mh1", "model_id": "fake-model"})
    store.put_samples("a", [
        {
            "run_id": "a", "row_id": "r1", "sample_idx": 1,
            "text": "<|x|>final answer", "prompt_tokens": 1, "gen_tokens": 3,
            "stop_reason": "stop", "verdict": 1.0, "extracted": "x",
            "failure_kind": None, "error": None, "est_cost_usd": 0.0,
            "token_ids": [1, 2, 3],
        }
    ])

    class _Fake:
        def decode(self, ids):
            return "<|content_thinking|>let me think<|message_model|><|x|>final answer"

    monkeypatch.setattr("flipbook.decode.tokenizer", lambda _m: _Fake())
    c = _client(tmp_path)
    d = c.get("/api/compare/row", params={"a": "a", "b": "b", "row": "r1"}).json()
    assert d["k_a"] == 2 and d["k_b"] == 1
    s = d["a"][1]
    assert s["text_clean"] == "final answer"
    assert s["thinking"] == "let me think"
    # the first sample has no token_ids
    assert d["a"][0]["thinking"] is None


def test_divergence_and_effort(tmp_path):
    _store(tmp_path)
    c = _client(tmp_path)
    d = c.get("/api/divergence", params={"base": "a", "ckpt": "b"}).json()
    assert d[0]["sum_nats"] == -0.8
    e = c.get("/api/effort", params={"run": "a", "pair": "0.2,0.9"}).json()
    assert e[0]["gap_nats"] == 20.0
    assert c.get("/api/divergence", params={"base": "a", "ckpt": "x"}).status_code == 404
    assert c.get("/api/effort", params={"run": "a", "pair": "bad"}).status_code == 400


def test_divergence_trace(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.put_run({"run_id": "a", "k": 1, "model_id": "fake-model"})
    store.put_samples("a", [
        {
            "run_id": "a", "row_id": "r1", "sample_idx": 1,
            "text": "hi", "prompt_tokens": 1, "gen_tokens": 2,
            "stop_reason": "stop", "verdict": 1.0, "extracted": "x",
            "failure_kind": None, "error": None, "est_cost_usd": 0.0,
            "token_ids": [5, 7],
        }
    ])
    store.put_divergence("a", "b", [
        {
            "row_id": "r1", "sample_idx": 1, "lp_base": [0.0, 0.0],
            "lp_ckpt": [-0.5, -1.0], "delta": [-0.5, -1.0],
            "sum_nats": -1.5, "mean_nats": -0.75,
            "divergence_pos": 0, "win_argmin": 0,
            "p_skip_base": 0.0, "p_skip_ckpt": 0.0, "cost_usd": 0.0,
        }
    ])

    class _Fake:
        def decode(self, ids):
            return f"<{ids[0]}>"

    monkeypatch.setattr("flipbook.api._tokenizer", lambda _m: _Fake())
    c = _client(tmp_path)
    t = c.get(
        "/api/divergence/trace",
        params={"base": "a", "ckpt": "b", "row": "r1", "sample": 1},
    ).json()
    assert t["tokens"] == [{"t": "<5>", "d": -0.5}, {"t": "<7>", "d": -1.0}]
    # r0:0 has divergence but no token_ids → honest 404, not a stub
    assert c.get(
        "/api/divergence/trace",
        params={"base": "a", "ckpt": "b", "row": "r0", "sample": 0},
    ).status_code == 404


def test_metrics_and_budget(tmp_path):
    _store(tmp_path)
    c = _client(tmp_path)
    assert c.get("/api/studies").json() == ["s1"]
    m = c.get("/api/metrics", params={"study": "s1"}).json()
    assert m[0]["key"] == "loss" and m[0]["value"] == 0.5
    b = c.get("/api/budget/a").json()
    assert b["n_samples"] == 4 and b["gen_p50"] == 50.0
    assert c.get("/api/metrics", params={"study": "nope"}).status_code == 404


def test_study_detail(tmp_path):
    store = _store(tmp_path)
    store.put_run({"run_id": "s1a", "study": "s1",
                   "provenance": {"train_step_measured": 8}})
    store.put_run({"run_id": "s1b", "study": "s1", "label": "baseline"})
    c = _client(tmp_path)
    d = c.get("/api/studies/s1").json()
    assert [r["run_id"] for r in d["runs"]] == ["s1b", "s1a"]
    assert d["runs"][1]["train_step"] == 8  # lifted from provenance
    assert d["metrics"]["loss"] == [{"step": 8, "value": 0.5}]
    assert d["effort_gap"] == {"a": 20.0}
    # a study with runs but no imported metrics still shows up
    assert c.get("/api/studies/nope").status_code == 404


def test_studies_union(tmp_path):
    store = _store(tmp_path)
    store.put_run({"run_id": "orphan", "study": "no_metrics"})
    c = _client(tmp_path)
    assert set(c.get("/api/studies").json()) == {"s1", "no_metrics"}


def test_load_env_file(tmp_path, monkeypatch):
    from flipbook.api import load_env_file

    f = tmp_path / "x.env"
    f.write_text(
        "# comment\n\nTINKER_API_KEY='tk_fake_value'\nexport OTHER=\"q\"\nALREADY=new\nbad line\n"
    )
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("ALREADY", "shell")
    assert load_env_file(f) == ["TINKER_API_KEY", "OTHER"]
    import os

    assert os.environ["TINKER_API_KEY"] == "tk_fake_value"
    assert os.environ["OTHER"] == "q"
    assert os.environ["ALREADY"] == "shell"  # shell wins over file
    assert load_env_file(tmp_path / "missing.env") == []


def test_study_charts_baseline_step(tmp_path):
    # a study with no imported training_metrics still gets a step-0 point
    # from the baseline's side of a live compare
    from flipbook.synthetic import seed_synthetic

    seed_synthetic(Store(tmp_path))
    c = _client(tmp_path)
    d = c.get("/api/studies/synthetic").json()
    steps = {p["step"] for p in d["metrics"]["eval/pass1"]}
    assert steps == {0, 24, 48, 64}
    step0 = next(p for p in d["metrics"]["eval/pass1"] if p["step"] == 0)
    assert abs(step0["value"] - 0.95) < 0.01


def test_study_synthesizes_into_the_evaluator_namespace(tmp_path):
    # an in-loop evaluator already wrote step 24 under its own prefix; the
    # synthesized points for the other checkpoints must join that series
    from flipbook.synthetic import seed_synthetic

    store = Store(tmp_path)
    seed_synthetic(store)
    store.put_training_metrics("synthetic", [
        {"study": "synthetic", "step": 24, "key": "flipbook/m/pass1", "value": 0.0}])
    d = _client(tmp_path).get("/api/studies/synthetic").json()
    assert "eval/pass1" not in d["metrics"]
    assert {p["step"] for p in d["metrics"]["flipbook/m/pass1"]} == {0, 24, 48, 64}


def test_study_does_not_chart_step0_twice(tmp_path):
    # a checkpoint trained for zero steps is the base model; the baseline
    # run must not be added as a second step-0 point on top of it
    from flipbook.synthetic import seed_synthetic

    store = Store(tmp_path)
    seed_synthetic(store)
    runs = store.runs()
    step24 = next(r for r in runs if r.get("train_step") == 24)
    store.put_run({**step24, "train_step": 0})
    d = _client(tmp_path).get("/api/studies/synthetic").json()
    steps = [p["step"] for p in d["metrics"]["eval/pass1"]]
    assert steps.count(0) == 1


def test_thinking_without_tokenizer():
    from flipbook.decode import thinking

    assert thinking("synthetic/Nope", [1, 2, 3], "x") is None



def test_study_charts_shared_baseline(tmp_path):
    # the baseline run belongs to another study; checkpoints reference it via
    # provenance.baseline_run_id and the study page still gets its step-0 arc
    from flipbook.synthetic import seed_synthetic

    store = Store(tmp_path)
    seed_synthetic(store)
    base = next(r for r in store.runs() if r.get("train_step") is None)
    base["study"] = "other"
    store.put_run(base)
    for r in store.runs("synthetic"):
        r.setdefault("provenance", {})["baseline_run_id"] = base["run_id"]
        store.put_run(r)
    c = _client(tmp_path)
    d = c.get("/api/studies/synthetic").json()
    assert base["run_id"] in {r["run_id"] for r in d["runs"]}
    assert {p["step"] for p in d["metrics"]["eval/pass1"]} == {0, 24, 48, 64}


def test_rollout_summary_averages_episodes_and_marks_thresholds(tmp_path):
    store = Store(tmp_path)
    rows = []
    for i, (a, b) in enumerate(zip([0.5, 0.6, 1, 1, 1, 1], [1, 1, 1, 1, 0.7, 0.6])):
        for traj in range(2):
            # A two-step episode and a one-step episode each get one vote.
            for step in range(traj + 1):
                rows.append({
                    "split": "train", "iteration": i, "group_idx": 0, "traj_idx": traj,
                    "step_idx": step, "ac_len": 10, "total_reward": -0.5,
                    "metrics": json.dumps(
                        {"a": a, "b": b, "n": 100} if step == traj else {"a": 0, "b": 0}),
                })
    store.put_rollouts("r", rows)
    c = _client(tmp_path)
    d = c.get("/api/rollouts/r").json()
    assert d["coverage"] == {"train": {"rollouts": 12, "texts": 0}}
    first = d["series"]["train"][0]
    assert first["tokens"] == 15 and first["n"] == 2 and first["reward"] == -0.5
    assert first["components"] == {"a": 0.5, "b": 1, "n": 100}
    events = d["events"]["train"]
    # a: trailing means .7, .867, 1 -> first saturation at iteration 4.
    # b: trailing means 1, 1, .9 -> first 10-point drop at iteration 4.
    assert {(e["component"], e["kind"], e["iteration"]) for e in events} == {
        ("a", "saturated", 4), ("b", "falling", 4)}
    assert next(e["to"] for e in events if e["component"] == "b") == pytest.approx(0.9)
    assert c.get("/api/rollouts/missing").status_code == 404
    assert "r" in c.get("/api/studies").json()
    assert c.get("/api/studies/r").json()["runs"] == []


def test_rollout_missing_metrics_are_not_zero_or_threshold_events(tmp_path):
    store = Store(tmp_path)
    store.put_rollouts("r", [
        {"split": "train", "iteration": i, "group_idx": 0, "traj_idx": t,
         "step_idx": 0, "ac_len": 10, "total_reward": 1,
         "metrics": json.dumps({"a": 1, "b": float("nan")} if i == 0 and t == 0 else {})}
        for i in range(4) for t in range(2)
    ])
    d = _client(tmp_path).get("/api/rollouts/r").json()
    assert d["series"]["train"][0]["components"] == {"a": 1}
    assert d["series"]["train"][0]["component_n"] == {"a": 1}
    assert d["series"]["train"][1]["components"] == {}
    assert d["events"]["train"] == []


def test_rollout_transcript_filters_sorting_and_coverage(tmp_path):
    store = Store(tmp_path)
    store.put_rollouts("r", [
        {"split": "train", "iteration": i, "group_idx": 0, "traj_idx": t,
         "step_idx": 0, "ac_len": 10, "total_reward": 1, "metrics": "{}"}
        for i in range(2) for t in range(4)
    ])
    store.put_rollout_texts("r", [
        {"split": split, "iteration": i, "group_pos": 0, "traj_pos": t,
         "prompt": json.dumps([
             {"role": "user", "content": "few-shot example"},
             {"role": "assistant", "content": "example answer"},
             {"role": "user", "content": [{"type": "text", "text": "actual question"}]},
         ]),
         "thinking": "x" * length, "answer": "answer",
         "reward_fields": json.dumps({"correct": correct, "reward": "1.0"})}
        for split, i, t, length, correct in [
            ("train", 0, 0, 10, "False"), ("train", 0, 1, 20, "False"),
            ("train", 0, 2, 30, "True"), ("test", 0, 0, 10, "True"),
        ]
    ])
    c = _client(tmp_path)
    d = c.get("/api/rollouts/r").json()
    assert d["coverage"]["train"] == {"rollouts": 8, "texts": 3}
    assert [p["n_texts"] for p in d["series"]["train"]] == [3, 0]
    assert d["text_fields"]["train"] == ["correct"]
    params = {"iteration": 0, "field": "correct", "value": "False", "limit": 1}
    d = c.get("/api/rollouts/r/texts", params={**params, "sort": "short"}).json()
    assert d["iterations"] == [0] and d["total"] == 2 and len(d["rows"]) == 1
    assert d["rows"][0]["traj_pos"] == 0 and d["rows"][0]["chars"] == 16
    assert d["rows"][0]["question"] == "actual question"
    d = c.get("/api/rollouts/r/texts", params={**params, "sort": "long"}).json()
    assert d["rows"][0]["traj_pos"] == 1
    assert c.get("/api/rollouts/r/texts?iteration=1").json()["rows"] == []
    assert c.get("/api/rollouts/r/texts?iteration=0&field=correct").status_code == 400
    assert c.get("/api/rollouts/r/texts?iteration=0&value=False").status_code == 400
    assert c.get("/api/rollouts/r/texts?iteration=0&limit=0").status_code == 422
    assert c.get("/api/rollouts/r/texts?iteration=0&sort=nope").status_code == 422
    assert c.get("/api/rollouts/ghost/texts?iteration=0").status_code == 404
