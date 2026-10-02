"""Frozen eval manifests and content-addressed row ids.

`row_id = "{benchmark}:{sha256(question)[:12]}"` where `question` is the last
user message: text the model sees, minus the system prefix. Hashing the
prompt-visible text (not a raw dataset field) keeps frozen and imported rows
consistent, and the same question under a different prompt template
gets a different id.
"""


import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "manifest/v2"

_SOURCES = {
    "gsm8k": "openai/gsm8k:main:test",
    "math500": "HuggingFaceH4/MATH-500:test",
    "aime2026": "MathArena/aime_2026",
}
_LICENSE = {
    "gsm8k": "MIT (openai/gsm8k)",
    "math500": "see HuggingFaceH4/MATH-500 dataset card",
    "aime2026": "see MathArena/aime_2026 dataset card",
}
# WHY: legacy rows key the benchmark off `grader`, not source_ids.benchmark
# (aime rows carry "aime_2026" there — the dataset name, not the benchmark name).
_LEGACY_BENCH = {"gsm8k": "gsm8k", "math500": "math500", "aime": "aime2026"}

_BOXED_SYS = {"role": "system", "content": "Put your final answer in \\boxed{}."}
_AIME_SUFFIX = (
    "\n\nThis is an AIME problem. The answer is an integer from 000 to 999. "
    "Show your work step by step, then put your final answer in \\boxed{}."
)


def question_text(messages: list[dict]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def make_row_id(benchmark: str, question: str) -> str:
    return f"{benchmark}:{hashlib.sha256(question.encode()).hexdigest()[:12]}"


def manifest_hash(rows: list[dict]) -> str:
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class Manifest:
    name: str
    manifest_hash: str
    spec: dict
    created_at: str
    rows: list[dict]

    @classmethod
    def freeze(cls, benchmarks: dict[str, int], seed: int, name: str) -> "Manifest":
        rows: list[dict] = []
        for bench, n in benchmarks.items():
            if bench not in _BUILDERS:
                raise ValueError(f"unknown benchmark {bench!r}; have {sorted(_BUILDERS)}")
            rows.extend(_BUILDERS[bench](n, seed))
        seen: set[str] = set()
        uniq = []
        for r in rows:
            if r["row_id"] in seen:
                continue
            seen.add(r["row_id"])
            uniq.append(r)
        if not uniq:
            raise ValueError("freeze produced zero rows")
        random.Random(seed).shuffle(uniq)
        spec = {
            "benchmarks": {
                b: {"n": n, "hf_path": _SOURCES[b], "license": _LICENSE[b]}
                for b, n in benchmarks.items()
            },
            "seed": seed,
        }
        return cls(
            name=name,
            manifest_hash=manifest_hash(uniq),
            spec=spec,
            created_at=datetime.now(UTC).isoformat(),
            rows=uniq,
        )

    @classmethod
    def import_v1(cls, path: str | Path, name: str | None = None) -> "Manifest":
        """Re-key a legacy manifest's rows; the old row_id moves to
        source_ids.legacy_row_id so old runs stay joinable."""
        doc = json.loads(Path(path).read_text())
        rows = doc["rows"]
        if doc.get("manifest_hash") != manifest_hash(rows):
            raise ValueError(f"{path}: legacy manifest_hash mismatch — file was modified")
        out = []
        for r in rows:
            bench = _LEGACY_BENCH[r["grader"]]
            out.append(
                {
                    "row_id": make_row_id(bench, question_text(r["messages"])),
                    "benchmark": bench,
                    "messages": r["messages"],
                    "gold": r["gold"],
                    "grader_id": r["grader"],
                    "source_ids": {**r["source_ids"], "legacy_row_id": r["row_id"]},
                }
            )
        return cls(
            name=name or Path(path).stem,
            manifest_hash=manifest_hash(out),
            spec={
                "imported_from": Path(path).name,
                "legacy_manifest_hash": doc["manifest_hash"],
                "benchmarks": doc["benchmarks"],
                "seed": doc["seed"],
            },
            created_at=datetime.now(UTC).isoformat(),
            rows=out,
        )

    @classmethod
    def load(cls, store: Any, name_or_hash: str) -> "Manifest":
        doc = store.manifest_doc(name_or_hash)
        if doc is None:
            raise KeyError(f"no manifest {name_or_hash!r}")
        return cls(
            name=doc["name"],
            manifest_hash=doc["manifest_hash"],
            spec=doc["spec"],
            created_at=doc["created_at"],
            rows=store.manifest_rows(doc["manifest_hash"]),
        )

    def to_doc(self) -> dict:
        return {
            "manifest_version": MANIFEST_VERSION,
            "manifest_hash": self.manifest_hash,
            "name": self.name,
            "n_rows": len(self.rows),
            "created_at": self.created_at,
            "spec": self.spec,
        }


def _row(benchmark: str, grader_id: str, messages: list[dict], gold: str, hf_path: str, idx: str) -> dict:
    return {
        "row_id": make_row_id(benchmark, question_text(messages)),
        "benchmark": benchmark,
        "messages": messages,
        "gold": gold,
        "grader_id": grader_id,
        "source_ids": {"benchmark": benchmark, "hf_path": hf_path, "idx": idx},
    }


def _rows_gsm8k(n: int, seed: int) -> list[dict]:
    from tinker_cookbook.eval.benchmarks._common import limit_dataset, load_benchmark_dataset

    ds = limit_dataset(load_benchmark_dataset("openai/gsm8k", name="main"), n, shuffle_seed=seed)
    return [
        _row(
            "gsm8k", "gsm8k",
            [_BOXED_SYS, {"role": "user", "content": row["question"]}],
            row["answer"].split("####")[-1].strip(),
            _SOURCES["gsm8k"], str(i),
        )
        for i, row in enumerate(ds)
    ]


def _rows_math500(n: int, seed: int) -> list[dict]:
    from tinker_cookbook.eval.benchmarks._common import limit_dataset, load_benchmark_dataset
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    ds = limit_dataset(load_benchmark_dataset("HuggingFaceH4/MATH-500"), n, shuffle_seed=seed)
    return [
        _row(
            "math500", "math500",
            [{"role": "user", "content": row["problem"] + " Put your final answer in \\boxed{}."}],
            str(extract_boxed(row["solution"])),
            _SOURCES["math500"], str(i),
        )
        for i, row in enumerate(ds)
    ]


def _rows_aime2026(n: int, seed: int) -> list[dict]:
    # AIME 2026 postdates plausible training cutoffs — the least-contaminated
    # benchmark in the set.
    from tinker_cookbook.eval.benchmarks._common import limit_dataset, load_benchmark_dataset

    ds = None
    for split in ("test", "train"):
        try:
            ds = load_benchmark_dataset("MathArena/aime_2026", split=split)
            break
        except Exception:  # noqa: BLE001, S112 — try the next split
            continue
    if ds is None:
        raise ValueError("could not load MathArena/aime_2026")
    ds = limit_dataset(ds, n, shuffle_seed=seed)
    rows = []
    for i, row in enumerate(ds):
        problem = row.get("problem") or row.get("question") or row.get("Problem", "")
        expected = str(row.get("answer") or row.get("Answer") or row.get("expected_answer", "")).strip()
        if not problem or not expected:
            continue
        rows.append(
            _row(
                "aime2026", "aime",
                [_BOXED_SYS, {"role": "user", "content": problem + _AIME_SUFFIX}],
                expected,
                _SOURCES["aime2026"], str(i),
            )
        )
    return rows


_BUILDERS = {"gsm8k": _rows_gsm8k, "math500": _rows_math500, "aime2026": _rows_aime2026}
