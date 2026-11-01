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
