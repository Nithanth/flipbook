"""Eval configuration and run identity.

"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunConfig:
    model: str  # base model name or tinker:// checkpoint path
    manifest_hash: str
    effort: float | None  # None for non-effort models
    temperature: float
    max_tokens: int
    k: int
    seed: int
    renderer: str
    study: str | None = None
    label: str | None = None


def canonical(cfg: RunConfig) -> dict:
    d = asdict(cfg)
    del d["label"]
    return d


def config_fp(cfg: RunConfig) -> str:
    blob = json.dumps(canonical(cfg), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def run_id(cfg: RunConfig) -> str:
    return config_fp(cfg)[:12]
