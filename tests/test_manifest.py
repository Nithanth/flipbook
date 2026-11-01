import json
import re

import pytest

from flipbook.manifest import (
    Manifest,
    make_row_id,
    manifest_hash,
    question_text,
)

import_v1 = Manifest.import_v1


def _row(question, benchmark="gsm8k", grader_id="gsm8k"):
    return {
        "row_id": make_row_id(benchmark, question),
        "benchmark": benchmark,
        "messages": [{"role": "user", "content": question}],
        "gold": "1",
        "grader_id": grader_id,
        "source_ids": {},
    }


def test_row_id_format_and_determinism():
    rid = make_row_id("gsm8k", "2+2?")
    assert re.fullmatch(r"gsm8k:[0-9a-f]{12}", rid)
    assert rid == make_row_id("gsm8k", "2+2?")
    assert rid != make_row_id("math500", "2+2?")


def test_hash_stable_across_dict_key_order():
    a = _row("q")
    b = {k: a[k] for k in reversed(list(a))}
    assert manifest_hash([a]) == manifest_hash([b])


def test_question_text_picks_last_user_message():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u2"},
    ]
    assert question_text(msgs) == "u2"


def _legacy_doc(rows):
    return {
        "manifest_version": "manifest/v1",
        "created_at": "2026-09-26T00:00:00+00:00",
        "benchmarks": ["gsm8k"],
        "n_requested": len(rows),
        "n_rows": len(rows),
        "seed": 42,
        "manifest_hash": manifest_hash(rows),
        "rows": rows,
    }


def test_import_v1_rekeys_and_records_legacy(tmp_path):
    legacy_rows = [
        {
            "row_id": "gsm8k_deadbeefcafe",
            "messages": [{"role": "user", "content": "2+2?"}],
            "gold": "4",
            "grader": "gsm8k",
            "source_ids": {"benchmark": "gsm8k", "hf_path": "openai/gsm8k:main:test", "idx": "0"},
        }
    ]
    p = tmp_path / "mix.json"
    p.write_text(json.dumps(_legacy_doc(legacy_rows)))

    m = import_v1(p)
    (r,) = m.rows
    assert re.fullmatch(r"gsm8k:[0-9a-f]{12}", r["row_id"])
    assert r["row_id"] != "gsm8k_deadbeefcafe"
    assert r["source_ids"]["legacy_row_id"] == "gsm8k_deadbeefcafe"
    assert r["source_ids"]["hf_path"] == "openai/gsm8k:main:test"
    assert m.spec["legacy_manifest_hash"] == manifest_hash(legacy_rows)
    assert m.name == "mix"


def test_import_v1_maps_aime_grader_to_benchmark(tmp_path):
    legacy_rows = [
        {
            "row_id": "aime2026_0123456789ab",
            "messages": [{"role": "user", "content": "Find x.\\n\\nThis is an AIME problem."}],
            "gold": "669",
            "grader": "aime",
            "source_ids": {"benchmark": "aime_2026", "hf_path": "MathArena/aime_2026", "idx": "0"},
        }
    ]
    p = tmp_path / "aime.json"
    p.write_text(json.dumps(_legacy_doc(legacy_rows)))
    m = import_v1(p)
    assert m.rows[0]["benchmark"] == "aime2026"
    assert m.rows[0]["row_id"].startswith("aime2026:")
    assert m.rows[0]["grader_id"] == "aime"


def test_import_v1_rejects_tampered(tmp_path):
    doc = _legacy_doc(
        [{"row_id": "x", "messages": [{"role": "user", "content": "q"}],
          "gold": "1", "grader": "gsm8k", "source_ids": {}}]
    )
    doc["manifest_hash"] = "0" * 64
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="mismatch"):
        import_v1(p)


def test_freeze_jsonl(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n".join([
        json.dumps({"question": "2+2?", "gold": "4", "grader_id": "exact"}),
        json.dumps({
            "messages": [{"role": "user", "content": "3+3?"}],
            "gold": "6", "grader_id": "number",
        }),
        json.dumps({"question": "2+2?", "gold": "4", "grader_id": "exact"}),  # dup
        json.dumps({"question": "5+5?", "gold": "10", "grader_id": "number",
                    "system": "Be terse."}),
    ]))
    m = Manifest.freeze_jsonl(p, benchmark="mine", seed=0, name="m")
    assert len(m.rows) == 3  # the duplicate question collapses
    r0 = m.rows[0]
    assert r0["row_id"] == make_row_id("mine", "2+2?")
    assert r0["benchmark"] == "mine"
    assert r0["gold"] == "4" and r0["grader_id"] == "exact"
    assert r0["source_ids"]["file"] == "tasks.jsonl" and r0["source_ids"]["line"] == 1
    assert m.rows[1]["row_id"] == make_row_id("mine", "3+3?")
    assert m.rows[2]["messages"][0] == {"role": "system", "content": "Be terse."}
    assert m.spec["benchmarks"] == {"mine": {"n": 3, "source": "tasks.jsonl"}}


def test_freeze_jsonl_errors_name_the_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        json.dumps({"question": "q", "gold": "1", "grader_id": "exact"}) + "\n"
        + json.dumps({"question": "q2", "grader_id": "exact"}) + "\n"
    )
    with pytest.raises(ValueError, match=":2:"):
        Manifest.freeze_jsonl(p, benchmark="b", seed=0, name="m")
    p.write_text(json.dumps({"gold": "1", "grader_id": "exact"}))
    with pytest.raises(ValueError, match=":1:"):
        Manifest.freeze_jsonl(p, benchmark="b", seed=0, name="m")


def test_manifest_roundtrip_shape():
    m = Manifest(
        name="m", manifest_hash="h", spec={}, created_at="t",
        rows=[_row("q1"), _row("q2")],
    )
    doc = m.to_doc()
    assert doc["n_rows"] == 2
    assert doc["manifest_version"].startswith("manifest/")
