import json

from flipbook.import_rollouts import import_rollouts
from flipbook.store import Store


def _summary(it, split, group, traj, steps, reward=1.0):
    return {
        "schema_version": 1, "split": split, "iteration": it,
        "group_idx": group, "traj_idx": traj, "tags": ["problems"],
        "sampling_client_step": it,
        "total_reward": reward, "final_reward": 0.0,
        "trajectory_metrics": {"env": {"x": 1}},
        "steps": [
            {"step_idx": i, "ob_len": 10 + i, "ac_len": 5 * (i + 1),
             "reward": reward / len(steps), "episode_done": done,
             "metrics": {"correct": 1.0, "short": 0.5}, "logs": {}}
            for i, done in enumerate(steps)
        ],
        "final_ob_len": 0,
    }


def _convo_div(messages):
    return {"tag": "div", "attrs": {"class": "lt-section-body"}, "children": [
        {"tag": "div", "attrs": {}, "children": [],
         "data": {"type": "conversation", "messages": messages}}]}


def _sec(title, body_children):
    return {"tag": "section", "attrs": {"class": "lt-section"}, "children": [
        {"tag": "h3", "attrs": {}, "children": [title]},
        {"tag": "div", "attrs": {"class": "lt-section-body"}, "children": body_children}]}


def _reward_sec(fields):
    return _sec("Reward", [{"tag": "table", "attrs": {}, "children": [
        {"tag": "tbody", "attrs": {}, "children": [
            {"tag": "tr", "attrs": {}, "children": [
                {"tag": "td", "attrs": {}, "children": [k]},
                {"tag": "td", "attrs": {}, "children": [v]}]}
            for k, v in fields.items()]}]}])


def _logtree():
    rollout1 = [
        _sec("Prompt", [_convo_div([{"role": "user", "content": "q1"}])]),
        _sec("Policy Response", [_convo_div([{
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "think1"},
                        {"type": "text", "text": "ans1"}]}])]),
        _reward_sec({"correct": "True", "reward": "1.000"}),
    ]
    rollout2 = [
        _sec("Prompt", [_convo_div([{"role": "user", "content": "q2"}])]),
        _sec("Policy Response", [_convo_div([{
            "role": "assistant", "content": "ans2"}])]),
        _reward_sec({"correct": "False", "reward": "0.000"}),
    ]
    group = {"tag": "section", "attrs": {"class": "lt-section"}, "children": [
        {"tag": "h2", "attrs": {}, "children": ["Group Rollout"]},
        {"tag": "div", "attrs": {}, "children": rollout1 + rollout2}]}
    return {"root": {"tag": "body", "attrs": {}, "children": [group]}}


def _seed_log_dir(path):
    for it in (0, 1):
        d = path / f"iteration_{it:06d}"
        d.mkdir(parents=True)
        recs = []
        for g in (0, 1):
            for t in (0, 1):
                steps = [False, True] if (it, g, t) == (0, 0, 0) else [True]
                recs.append(_summary(it, "train", g, t, steps))
        (d / "train_rollout_summaries.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n")
    (path / "iteration_000000" / "train_logtree.json").write_text(
        json.dumps(_logtree()))
    (path / "iteration_000001" / "eval_test_rollout_summaries.jsonl").write_text(
        json.dumps(_summary(1, "test", 0, 0, [True])) + "\n")


def test_import_rollouts(tmp_path):
    log = tmp_path / "log"
    _seed_log_dir(log)
    store = Store(tmp_path / "store")
    imp = import_rollouts(store, log, "s")
    assert imp.splits == ["test", "train"]
    assert imp.iterations == 2
    assert imp.n_rollouts == 9  # 4 + 4 train, 1 test
    assert imp.n_steps == 10    # one train traj has 2 steps
    assert imp.n_texts == 2

    rows = store.rollouts("s").to_pylist()
    assert len(rows) == 10
    multi = [r for r in rows if r["n_steps"] == 2]
    assert len(multi) == 2
    assert {r["step_idx"] for r in multi} == {0, 1}
    assert all(r["split"] == "train" and r["iteration"] == 0
               and r["group_idx"] == 0 and r["traj_idx"] == 0 for r in multi)
    assert json.loads(multi[0]["metrics"]) == {"correct": 1.0, "short": 0.5}
    assert json.loads(multi[0]["trajectory_metrics"]) == {"env": {"x": 1}}
    assert rows[0]["policy_step"] == 0

    txts = store.rollout_texts("s").to_pylist()
    assert len(txts) == 2
    by_pos = {t["traj_pos"]: t for t in txts}
    assert by_pos[0]["thinking"] == "think1" and by_pos[0]["answer"] == "ans1"
    assert by_pos[1]["thinking"] == "" and by_pos[1]["answer"] == "ans2"
    assert json.loads(by_pos[0]["prompt"]) == [{"role": "user", "content": "q1"}]
    assert json.loads(by_pos[0]["reward_fields"])["reward"] == "1.000"

    # 2 of 4 train trajectories logged in iter 0, none in iter 1
    assert imp.text_coverage["train"] == (0.0, 0.5)
    assert imp.text_coverage["test"] == (0.0, 0.0)

    # re-import is idempotent
    imp2 = import_rollouts(store, log, "s")
    assert imp2.n_rollouts == imp.n_rollouts
    assert store.rollouts("s").num_rows == 10
    assert store.rollout_texts("s").num_rows == 2


def test_unrecognized_logtree_yields_no_texts(tmp_path):
    log = tmp_path / "log"
    d = log / "iteration_000000"
    d.mkdir(parents=True)
    (d / "train_rollout_summaries.jsonl").write_text(
        json.dumps(_summary(0, "train", 0, 0, [True])) + "\n")
    (d / "train_logtree.json").write_text(
        json.dumps({"root": {"tag": "body", "children": [
            {"tag": "section", "children": [
                {"tag": "h2", "children": ["Something Else"]}]}]}}))
    store = Store(tmp_path / "store")
    imp = import_rollouts(store, log, "s")
    assert imp.n_rollouts == 1 and imp.n_texts == 0
    assert store.rollout_texts("s").num_rows == 0
