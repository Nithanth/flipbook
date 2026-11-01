"""Pure config checks that catch silently-wrong evals before they cost money.

"""


from dataclasses import dataclass
from typing import Literal

from flipbook.config import RunConfig
from flipbook.pricing import PRICES


@dataclass(frozen=True)
class Finding:
    level: Literal["error", "warn"]
    code: str
    message: str


def lint(cfg: RunConfig, base_model: str | None = None) -> list[Finding]:
    model = base_model or (None if cfg.model.startswith("tinker://") else cfg.model)
    inkling = model is not None and model.startswith("thinkingmachines/Inkling")
    tml = cfg.renderer == "tml_v0"
    findings: list[Finding] = []

    if inkling and not tml:
        findings.append(
            Finding(
                "error", "E_INKLING_RENDERER",
                f"{model} requires the tml_v0 renderer; effort conditioning "
                "is mandatory and will be silently absent",
            )
        )
    if inkling and cfg.effort is None:
        findings.append(
            Finding(
                "error", "E_INKLING_NO_EFFORT",
                "Inkling requires an effort value in [0, 1)",
            )
        )
    if cfg.effort is not None and not 0.0 <= cfg.effort < 1.0:
        findings.append(
            Finding(
                "error", "E_EFFORT_RANGE",
                f"effort={cfg.effort} outside [0, 1)",
            )
        )
    if cfg.effort is not None and not tml:
        findings.append(
            Finding(
                "error", "E_EFFORT_NON_TML",
                f"effort={cfg.effort} set but renderer is {cfg.renderer!r}; "
                "effort only exists on tml_v0 and will be ignored",
            )
        )
    if model is not None and model not in PRICES:
        findings.append(
            Finding(
                "warn", "W_NO_PRICES",
                f"no price table for {model}; costs will be null",
            )
        )
    e, mt = cfg.effort, cfg.max_tokens
    if e is not None and ((e >= 0.7 and mt < 8192) or (e >= 0.9 and mt < 16384)):
        findings.append(
            Finding(
                "warn", "W_BUDGET",
                f"effort {e} with max_tokens={mt}: observed p99 generation at "
                "this effort exceeds the budget on hard tasks; failures will "
                "show as truncation, not wrong answers. Check `flipbook "
                "budget` on a completed run to size it",
            )
        )
    if cfg.temperature > 0 and cfg.k == 1:
        findings.append(
            Finding(
                "warn", "W_K1_STOCHASTIC",
                "temperature > 0 with k=1: verdicts are stochastic; pass@1 "
                "estimates will be noisy",
            )
        )
    return findings
