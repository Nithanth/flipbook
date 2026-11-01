"""Benchmark-native answer grading: `grade(grader_id, text, gold) -> Grade`.

"""


from dataclasses import dataclass

from tinker_cookbook.eval.benchmarks._common import (
    check_gsm8k,
    extract_boxed,
    extract_gsm8k_answer,
    extract_number,
)
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed as extract_last_boxed,
)
from tinker_cookbook.recipes.math_rl.math_grading import grade_answer


@dataclass(frozen=True)
class Grade:
    verdict: float
    extracted: str | None
    note: str | None = None


def grade(grader_id: str, text: str, gold: str) -> Grade:
    if grader_id == "gsm8k":
        extracted = extract_gsm8k_answer(text) or None
        return Grade(float(check_gsm8k(text, gold)), extracted, _note(extracted))
    if grader_id == "math500":
        try:
            extracted = extract_last_boxed(text) or None
        except ValueError:
            extracted = None
        verdict = grade_answer(extracted, gold) if extracted is not None else False
        return Grade(float(verdict), extracted, _note(extracted))
    if grader_id == "aime":
        boxed = extract_boxed(text)
        extracted = (extract_number(boxed) if boxed else extract_gsm8k_answer(text)) or None
        try:
            verdict = int(float(extracted)) == int(gold)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            verdict = False
        return Grade(float(verdict), extracted, _note(extracted))
    raise ValueError(f"unknown grader {grader_id!r}")


def _note(extracted: str | None) -> str | None:
    return None if extracted is not None else "unparsed"
