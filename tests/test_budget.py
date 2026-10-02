from flipbook.budget import budget
from flipbook.store import Store


def test_budget_report(tmp_path):
    store = Store(tmp_path)
    rows = [
        {
            "run_id": "r1", "row_id": f"a:{i}", "sample_idx": 0,
            "gen_tokens": g, "prompt_tokens": 10, "est_cost_usd": 0.001,
            "failure_kind": "truncation" if g == 1000 else None,
        }
        for i, g in enumerate([10, 20, 30, 1000])
    ]
    store.put_samples("r1", rows)

    b = budget(store, "r1")
    assert b.n_samples == 4
    assert b.n_truncated == 1
    assert b.gen_max == 1000
    assert b.total_gen_tokens == 1060
    assert b.total_prompt_tokens == 40
    assert abs(b.est_cost_usd - 0.004) < 1e-9
    assert b.gen_p50 in (10, 20, 30)


def test_budget_empty_run(tmp_path):
    b = budget(Store(tmp_path), "nope")
    assert b.n_samples == 0
    assert b.est_cost_usd is None
