import os
import subprocess
import sys
import textwrap
from pathlib import Path

from flipbook.manifest import Manifest, manifest_hash
from flipbook.store import Store

ROOT = Path(__file__).resolve().parents[1]


def _manifest(tmp_path, name="m"):
    rows = [
        {
            "row_id": f"gsm8k:{i:012d}",
            "benchmark": "gsm8k",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "gold": "1",
            "grader_id": "gsm8k",
            "source_ids": {},
        }
        for i in range(3)
    ]
    return Manifest(
        name=name,
        manifest_hash=manifest_hash(rows),
        spec={"seed": 42},
        created_at="t",
        rows=rows,
    )


def _sample(run_id, row_id, idx=0, **kw):
    return {"run_id": run_id, "row_id": row_id, "sample_idx": idx,
            "text": "t", "verdict": 1.0, **kw}


def test_manifest_roundtrip(tmp_path):
    store = Store(tmp_path)
    m = _manifest(tmp_path)
    store.put_manifest(m.to_doc(), m.rows)

    doc = store.manifest_doc("m")
    assert doc["manifest_hash"] == m.manifest_hash
    assert store.manifest_doc(m.manifest_hash[:12])["name"] == "m"
    assert store.manifest_doc("nope") is None

    rows = store.manifest_rows(m.manifest_hash)
    assert rows == m.rows  # messages/source_ids JSON round-trips


def test_run_lifecycle(tmp_path):
    store = Store(tmp_path)
    store.put_run({"run_id": "r1", "study": "s1"})
    store.put_run({"run_id": "r2", "study": "s2"})
    assert [r["run_id"] for r in store.runs()] == ["r1", "r2"]
    assert [r["run_id"] for r in store.runs("s1")] == ["r1"]


def test_resolve_run(tmp_path):
    store = Store(tmp_path)
    store.put_run({"run_id": "abcd1234ffff", "study": "s1", "label": "base"})
    store.put_run({"run_id": "abcd5678eeee", "study": "s1", "label": "ckpt"})
    store.put_run({"run_id": "99990000aaaa", "study": "s2", "label": "base"})

    assert store.resolve_run("abcd1234ffff")["label"] == "base"          # exact id
    assert store.resolve_run("abcd1234")["run_id"] == "abcd1234ffff"    # unique prefix
    assert store.resolve_run("s1/base")["run_id"] == "abcd1234ffff"     # study/label
    assert store.resolve_run("ckpt")["run_id"] == "abcd5678eeee"        # unique bare label

    try:
        store.resolve_run("base")  # label exists in two studies
    except LookupError as e:
        assert "ambiguous" in str(e) and "s1" in str(e)
    else:
        raise AssertionError("expected ambiguous label to fail")

    try:
        store.resolve_run("nope")
    except LookupError as e:
        assert "no run matches" in str(e)
    else:
        raise AssertionError("expected unknown ref to fail")


def test_put_run_warns_on_label_clash(tmp_path):
    import pytest
    store = Store(tmp_path)
    store.put_run({"run_id": "r1", "study": "s1", "label": "x"})
    store.put_run({"run_id": "r2", "study": "s2", "label": "x"})  # other study: fine
    with pytest.warns(UserWarning, match="already has label"):
        store.put_run({"run_id": "r3", "study": "s1", "label": "x"})
    store.put_run({"run_id": "r1", "study": "s1", "label": "x"})  # self: no warn


def test_samples_idempotent(tmp_path):
    store = Store(tmp_path)
    store.put_samples("r1", [_sample("r1", "a:1"), _sample("r1", "a:2")])
    assert store.has("r1", "a:1", 0)
    assert not store.has("r1", "nope", 0)

    store.put_samples("r1", [_sample("r1", "a:1"), _sample("r1", "a:3")])
    t = store.samples("r1")
    assert t.num_rows == 3  # a:1 deduped, a:3 appended
    assert store.has("r1", "a:3", 0)


def test_samples_missing_keys_become_null(tmp_path):
    store = Store(tmp_path)
    store.put_samples("r1", [{"run_id": "r1", "row_id": "a:1", "sample_idx": 0}])
    row = store.samples("r1").to_pylist()[0]
    assert row["verdict"] is None
    assert row["token_ids"] is None


def test_empty_samples(tmp_path):
    assert Store(tmp_path).samples("r1").num_rows == 0


def test_query_glob(tmp_path):
    store = Store(tmp_path)
    store.put_samples("r1", [_sample("r1", "a:1"), _sample("r1", "a:2")])
    store.put_samples("r2", [_sample("r2", "a:1")])
    t = store.query(
        "select count(*) c from read_parquet($p)", p=str(tmp_path / "samples" / "*.parquet")
    )
    assert t["c"][0].as_py() == 3


def test_reader_during_concurrent_writer(tmp_path):
    store = Store(tmp_path)
    store.put_samples("seed", [_sample("seed", "a:0")])
    writer = textwrap.dedent(
        """
        import sys
        from flipbook.store import Store
        s = Store(sys.argv[1])
        for i in range(10):
            s.put_samples(f"w{i}", [{"run_id": f"w{i}", "row_id": f"b:{i}",
                                     "sample_idx": 0, "text": "x"}])
        """
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    p = subprocess.Popen(
        [sys.executable, "-c", writer, str(tmp_path)], cwd=ROOT, env=env
    )
    try:
        for _ in range(50):
            t = store.query(
                "select count(*) c from read_parquet($p)",
                p=str(tmp_path / "samples" / "*.parquet"),
            )
            assert t["c"][0].as_py() >= 1
    finally:
        p.wait(timeout=30)
    assert p.returncode == 0
    t = store.query(
        "select count(*) c from read_parquet($p)", p=str(tmp_path / "samples" / "*.parquet")
    )
    assert t["c"][0].as_py() == 11


def test_no_torn_writes(tmp_path):
    store = Store(tmp_path)
    store.put_samples("r1", [_sample("r1", "a:1")])
    assert not list(tmp_path.glob("**/*.tmp"))
