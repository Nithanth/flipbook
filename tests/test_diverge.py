from flipbook.diverge import _pick_trace, _pos, _win_argmin
from flipbook.store import Store


def test_divergence_pos_crosses_tau():
    # cumsum hits -5 at index 2, recovers — first crossing wins
    assert _pos([0.0, -2.0, -3.5, 9.0], 5.0) == 2
    assert _pos([0.0, -1.0, -1.0], 5.0) is None


def test_win_argmin():
    delta = [0.0] * 8 + [-1.0] * 16 + [0.0] * 8
    pos = _win_argmin(delta, 16)
    assert pos is not None and 8 <= pos <= 24
    assert _win_argmin([0.0] * 5, 16) is None


def test_pick_trace_prefers_lowest_nonerror_idx():
    samples = [
        {"sample_idx": 1, "failure_kind": "wrong_answer", "token_ids": [5, 6]},
        {"sample_idx": 0, "failure_kind": "error", "token_ids": [1, 2]},
        {"sample_idx": 2, "failure_kind": None, "token_ids": [7, 8]},
    ]
    assert _pick_trace(samples)["sample_idx"] == 1
    assert _pick_trace([]) is None
    assert _pick_trace([{"sample_idx": 0, "failure_kind": None, "token_ids": None}]) is None


def test_divergence_store_roundtrip_and_resume(tmp_path):
    s = Store(tmp_path)
    row = {
        "base_run_id": "a", "ckpt_run_id": "b", "row_id": "r0", "sample_idx": 0,
        "prompt_len": 10, "n": 3, "lp_base": [-1.0, -2.0, -3.0],
        "lp_ckpt": [-1.5, -2.0, -3.0], "delta": [-0.5, 0.0, 0.0],
        "sum_nats": -0.5, "mean_nats": -0.166, "divergence_pos": None,
        "win_argmin": None, "p_skip_base": 0.01, "p_skip_ckpt": 0.9, "cost_usd": 0.001,
    }
    s.put_divergence("a", "b", [row])
    s.put_divergence("a", "b", [row])  # re-put dedupes, doesn't append
    t = s.divergence("a", "b").to_pylist()
    assert len(t) == 1 and t[0]["delta"] == [-0.5, 0.0, 0.0]
    assert s.divergence("a", "zzz").num_rows == 0


def test_effort_store_roundtrip(tmp_path):
    s = Store(tmp_path)
    row = {
        "run_id": "a", "row_id": "r0", "sample_idx": 0, "e_low": 0.2, "e_high": 0.9,
        "lp_low_sum": -10.0, "lp_high_sum": -4.0, "gap_nats": 6.0, "n": 5, "cost_usd": 0.001,
    }
    s.put_effort("a", 0.2, 0.9, [row])
    s.put_effort("a", 0.2, 0.9, [row])
    t = s.effort("a", 0.2, 0.9).to_pylist()
    assert len(t) == 1 and t[0]["gap_nats"] == 6.0
    assert s.effort("a", 0.0, 0.9).num_rows == 0
