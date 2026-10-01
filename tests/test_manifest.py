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


def test_manifest_roundtrip_shape():
    m = Manifest(
        name="m", manifest_hash="h", spec={}, created_at="t",
        rows=[_row("q1"), _row("q2")],
    )
    doc = m.to_doc()
    assert doc["n_rows"] == 2
    assert doc["manifest_version"].startswith("manifest/")
