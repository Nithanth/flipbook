from flipbook.store import Store
from flipbook.synthetic import seed_synthetic


def test_reseed_replaces_legacy_runs_and_artifacts(tmp_path):
    store = Store(tmp_path)
    seed_synthetic(store)
    for run in store.runs("synthetic"):
        rid = run["run_id"]
        legacy = rid.ljust(12, "0")
        if rid == legacy:
            continue
        store.put_run({**run, "run_id": legacy, "label": f"old-{run['label']}"})
        store.put_samples(legacy, [
            {**s, "run_id": legacy} for s in store.samples(rid).to_pylist()])
        (tmp_path / "runs" / f"{rid}.json").unlink()
        store.put_run({**run, "run_id": legacy})
        (tmp_path / "samples" / f"{rid}.parquet").unlink()
    old_trace = tmp_path / "divergence" / "syn_baseline__syn_step2400.parquet"
    (tmp_path / "divergence" / "syn_baseline__syn_step24.parquet").rename(old_trace)
    store.put_run({"run_id": "real", "study": "other", "label": "step24"})
    seed_synthetic(store)
    assert len(store.runs("synthetic")) == 4
    for label in ("baseline", "step24", "step48", "final"):
        run = store.resolve_run(f"synthetic/{label}")
        assert run["run_id"] == f"syn_{label}"
        assert store.samples(run["run_id"]).num_rows == 20
    assert not old_trace.exists()
    assert not list((tmp_path / "samples").glob("syn_*00.parquet"))
    assert store.resolve_run("real")["study"] == "other"
    assert store.divergence("syn_baseline", "syn_step24").num_rows == 10
