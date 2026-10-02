from flipbook.stats import compare, gate
from flipbook.store import Store


def _run(store, rid, verdicts, k=1, gen=100, stop="stop"):
    store.put_run({"run_id": rid, "k": k})
    store.put_samples(rid, [
        {
            "run_id": rid, "row_id": f"r{i}", "sample_idx": j,
            "text": "t", "prompt_tokens": 10, "gen_tokens": gen,
            "stop_reason": stop,
            "verdict": None if v is None else float(v),
            "extracted": "x" if v else None,
            "failure_kind": (
                "error" if v is None
                else None if v
                else "truncation" if stop == "length" else "wrong_answer"
            ),
            "error": "boom" if v is None else None,
            "est_cost_usd": 0.001,
        }
        for i, vs in verdicts.items() for j, v in enumerate(vs)
    ])


def test_k1_reduces_to_2x2(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1], 1: [1], 2: [0], 3: [1]})
    _run(store, "b", {0: [0], 1: [1], 2: [0], 3: [0]})
    p = compare(store, "a", "b")
    assert p.n_pairs == 4 and p.excluded == []
    assert p.acc_a == 0.75 and p.acc_b == 0.25
    assert p.delta == -0.5
    assert p.agreement == {"both_right": 1, "both_wrong": 1, "a_only": 2, "b_only": 0}
    assert [f["row_id"] for f in p.flips] == ["r0", "r3"]
    assert all(f["kind"] == "regression" and f["hard"] for f in p.flips)


def test_ci_excludes_zero_on_strong_regression(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {i: [1] for i in range(8)})
    _run(store, "b", {i: [0] for i in range(7)} | {7: [1]})
    p = compare(store, "a", "b")
    assert p.delta_ci[1] < 0
    ok, reasons = gate(p)
    assert not ok and any("delta" in r for r in reasons)


def test_identical_runs_gate_passes(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {i: [1] for i in range(6)})
    _run(store, "b", {i: [1] for i in range(6)})
    p = compare(store, "a", "b")
    assert p.delta == 0.0
    ok, reasons = gate(p)
    assert ok and reasons == []


def test_exclusions_named(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1], 1: [1], 2: [None], 3: [None]})
    _run(store, "b", {1: [1], 2: [1], 3: [None]})
    p = compare(store, "a", "b")
    reasons = {e["row_id"]: e["reason"] for e in p.excluded}
    assert p.n_pairs == 1
    assert reasons == {"r0": "missing_in_b", "r2": "errors", "r3": "errors_in_both"}


def test_k2_partial_rates_and_threshold(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1, 1], 1: [1, 0]}, k=2)
    _run(store, "b", {0: [0, 0], 1: [1, 0]}, k=2)
    p = compare(store, "a", "b")
    # r0: 1.0 -> 0.0 hard regression; r1: 0.5 -> 0.5 is a majority-tie, not a flip
    assert p.n_pairs == 2 and len(p.flips) == 1
    assert p.flips[0]["row_id"] == "r0" and p.flips[0]["hard"]
    assert p.acc_a == 0.75 and p.acc_b == 0.25


def test_taxonomy_and_token_stats(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1], 1: [0]}, gen=200)
    _run(store, "b", {0: [0], 1: [0]}, gen=50, stop="length")
    p = compare(store, "a", "b")
    assert p.truncation_rate_a == 0.0 and p.truncation_rate_b == 1.0
    assert p.failures["b"]["truncation"] == 2
    assert p.tokens["a"]["mean"] == 200 and p.tokens["b"]["mean"] == 50
    assert p.delta_tokens == -150
    ok, reasons = gate(p, max_new_truncation_rate=0.5)
    assert not ok and any("truncation" in r for r in reasons)


def test_no_pairs_is_empty_not_crash(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1]})
    _run(store, "b", {1: [1]})
    p = compare(store, "a", "b")
    assert p.n_pairs == 0 and p.delta == 0.0
    ok, _ = gate(p)
    assert ok
