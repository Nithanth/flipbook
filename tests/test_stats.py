import math

import pytest

from flipbook.stats import comparability, compare, gate
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
    assert p.passn == {}  # k=1 supports no n>=2


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


def test_pass_at_n(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1, 1, 0, 0], 1: [0, 0, 0, 0]}, k=4)
    _run(store, "b", {0: [1, 1, 1, 0], 1: [1, 0, 0, 0]}, k=4)
    p = compare(store, "a", "b")
    # pass@1 is still plain per-row mean correctness
    assert p.acc_a == 0.25 and p.acc_b == 0.5
    assert set(p.passn) == {"2", "3", "4"}
    # r0 in a: 4 graded, 2 correct -> pass@2 = 1 - C(2,2)/C(4,2) = 5/6; r1: 0
    assert p.passn["2"]["a"] == pytest.approx((1 - math.comb(2, 2) / math.comb(4, 2)) / 2)
    # b: r0 -> 1 - C(1,2)/C(4,2) = 1; r1 -> 1 - C(3,2)/C(4,2) = 1/2
    assert p.passn["2"]["b"] == pytest.approx(0.75)
    assert p.passn["2"]["delta"] == pytest.approx(1 / 3)
    lo, hi = p.passn["2"]["delta_ci"]
    assert lo <= p.passn["2"]["delta"] <= hi


def test_pass_at_n_uses_smaller_k(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1, 1], 1: [0, 1]}, k=2)
    _run(store, "b", {0: [0, 0, 0, 0], 1: [0, 0, 0, 1]}, k=4)
    p = compare(store, "a", "b")
    assert set(p.passn) == {"2"}  # capped by a's k=2
    assert p.passn["2"]["a"] == 1.0
    # b: r0 -> 0; r1 -> 1 - C(3,2)/C(4,2) = 1/2
    assert p.passn["2"]["b"] == pytest.approx(0.25)


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


def test_cells_cover_every_row(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1], 1: [1], 2: [0], 3: [1], 4: [1]})
    _run(store, "b", {0: [0], 1: [1], 2: [0], 3: [0]})
    p = compare(store, "a", "b")
    assert len(p.cells) == p.n_pairs + len(p.excluded)
    assert [c["row_id"] for c in p.cells] == sorted(c["row_id"] for c in p.cells)
    for kind, n in p.agreement.items():
        assert sum(c["cell"] == kind for c in p.cells) == n
    ex = [c for c in p.cells if c["cell"] == "excluded"]
    assert [c["row_id"] for c in ex] == ["r4"]
    assert ex[0]["p_a"] is None and ex[0]["p_b"] is None
    # wrong-answer failures in b carry the rows where those samples live
    assert set(p.failure_rows_b) <= set(p.failures["b"])
    assert p.failure_rows_b["wrong_answer"] == ["r0", "r2", "r3"]


def test_no_pairs_is_empty_not_crash(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1]})
    _run(store, "b", {1: [1]})
    p = compare(store, "a", "b")
    assert p.n_pairs == 0 and p.delta == 0.0
    ok, _ = gate(p)
    assert ok


FULL = {
    "manifest_hash": "m" * 64, "grader_id": "g", "model_id": "thinkingmachines/Inkling",
    "renderer": "tml_v0", "effort": 0.9, "k": 4, "temperature": 1.0,
    "max_tokens": 4096, "seed": 0,
}


def test_comparability_identical():
    c = comparability(FULL, dict(FULL))
    assert c == {"ok": True, "blocks": [], "warnings": [], "token_views": True}


def test_comparability_manifest_blocks():
    c = comparability(FULL, {**FULL, "manifest_hash": "n" * 64})
    assert not c["ok"] and "manifests" in c["blocks"][0]
    assert "mmmmmmmm…" in c["blocks"][0] and "nnnnnnnn…" in c["blocks"][0]


def test_comparability_model_warns_but_ok():
    c = comparability(FULL, {**FULL, "model_id": "thinkingmachines/Inkling-Small"})
    assert c["ok"] and c["blocks"] == []
    assert any("different base models" in w for w in c["warnings"])
    assert c["token_views"]  # same renderer, so the token gap still means something


def test_comparability_sampling_and_k_warnings():
    c = comparability(FULL, {**FULL, "effort": 0.5, "temperature": 0.0, "k": 2})
    assert c["ok"]
    assert any("effort 0.9 vs 0.5" in w and "temperature 1.0 vs 0.0" in w for w in c["warnings"])
    assert any("k differs (4 vs 2)" in w for w in c["warnings"])


def test_comparability_renderer_disables_token_views():
    c = comparability(FULL, {**FULL, "renderer": "qwen3"})
    assert c["ok"] and not c["token_views"]
    assert not comparability({**FULL, "renderer": None}, FULL)["token_views"]


def test_comparability_skips_missing_keys():
    c = comparability({"run_id": "a", "k": 1}, {"run_id": "b", "k": 1, "manifest_hash": "x"})
    assert c["ok"] and c["warnings"] == []


def test_compare_carries_comparability(tmp_path):
    store = Store(tmp_path)
    _run(store, "a", {0: [1]})
    _run(store, "b", {0: [1]})
    store.put_run({"run_id": "b", "k": 2})  # k differs → warning, still ok
    p = compare(store, "a", "b")
    assert p.comparability["ok"]
    assert any("k differs" in w for w in p.comparability["warnings"])
