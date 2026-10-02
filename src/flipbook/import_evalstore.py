"""Import a legacy eval bundle (runbundle/v1) into the store as a real run.
The bundle's manifest_hash is legacy-format; the run is re-keyed onto the
imported manifest by matching rows through source_ids.legacy_row_id. The
run's identity is a synthesized RunConfig
"""


import json
from pathlib import Path

from flipbook.config import RunConfig, run_id
from flipbook.runner import _failure_kind
from flipbook.store import Store


def _manifest_for(store: Store, legacy_ids: set[str]) -> dict:
    """Find the stored manifest whose rows cover the bundle's legacy ids."""
    for doc in sorted(store.path.glob("manifests/*.json"), key=lambda p: p.name):
        d = json.loads(doc.read_text())
        rows = store.manifest_rows(d["manifest_hash"])
        legacy = {r["source_ids"].get("legacy_row_id"): r["row_id"] for r in rows}
        if legacy_ids <= legacy.keys():
            return {"doc": d, "idmap": legacy}
    raise SystemExit("no stored manifest covers this bundle — import its manifest first")


def import_evalstore(store: Store, path: str | Path) -> str:
    bundle = json.loads(Path(path).read_text())
    cfg0 = bundle["config"]
    dec = cfg0["decoding"]
    rows = bundle["rows"]
    m = _manifest_for(store, {r["row_id"] for r in rows})
    idmap = m["idmap"]
    k = max(len(r["outputs"]) for r in rows)
    cfg = RunConfig(
        model=cfg0["checkpoint_path"] or cfg0["model_id"],
        manifest_hash=m["doc"]["manifest_hash"],
        effort=cfg0.get("effort"), temperature=dec.get("temperature", 0.0),
        max_tokens=dec.get("max_tokens", 32768), k=k, seed=dec.get("seed") or 0,
        renderer=cfg0["renderer"], study=bundle.get("provenance", {}).get("study"),
        label=cfg0.get("label"),
    )
    rid = run_id(cfg)
    samples = []
    for r in rows:
        for i, o in enumerate(r["outputs"]):
            verdict = None if o.get("error") else (o.get("grader") or {}).get("verdict")
            extracted = o.get("final_answer")
            samples.append(
                {
                    "run_id": rid, "row_id": idmap[r["row_id"]], "sample_idx": i,
                    "text": o.get("raw_output"), "prompt_tokens": o.get("prompt_tokens"),
                    "gen_tokens": o.get("gen_tokens"), "stop_reason": o.get("stop_reason"),
                    "verdict": verdict, "extracted": extracted,
                    "failure_kind": _failure_kind(verdict or 0.0, extracted, o.get("stop_reason") or "")
                    if o.get("error") is None else "error",
                    "grade_note": (o.get("grader") or {}).get("rationale"),
                    "error": o.get("error"), "est_cost_usd": o.get("est_cost_usd"),
                    "token_ids": None, "token_logprobs": o.get("token_logprobs"),
                }
            )
    store.put_samples(rid, samples)
    store.put_run(
        {
            "run_id": rid, "created_at": bundle.get("created_at"),
            "manifest_hash": m["doc"]["manifest_hash"],
            "study": cfg.study, "source": "import_evalstore",
            "label": cfg.label, "model_id": cfg0["model_id"],
            "checkpoint_path": cfg0.get("checkpoint_path"),
            "base_model": cfg0.get("base_model") or cfg0["model_id"],
            "train_step": cfg0.get("train_step"),
            "renderer": cfg0["renderer"], "effort": cfg0.get("effort"),
            "temperature": dec.get("temperature"), "max_tokens": dec.get("max_tokens"),
            "k": k, "seed": dec.get("seed") or 0,
            "grader_id": "benchmark-native", "grader_version": cfg0.get("grader_version"),
            "provenance": {
                "imported_from": Path(path).name,
                "legacy_run_id": bundle.get("run_id"),
                "legacy_config_hash": bundle.get("config_hash"),
            },
        }
    )
    return rid
