"""Benchmark-native answer grading: `grade(grader_id, text, gold) -> Grade`.

GRADERS maps a short id to (fn, one-line description). Two dynamic forms
resolve outside the registry: `regex:<pattern>` and `module.path:func`.
"""


import importlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from tinker_cookbook.eval.benchmarks._common import (
    check_gsm8k,
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


class UnknownGraderError(ValueError):
    """grader_id is not in GRADERS and matches no dynamic form."""


GraderFn = Callable[[str, str], Grade]


_CONTROL_TOKEN = re.compile(r"<\|[^|>]+\|>")


def strip_control_tokens(text: str) -> str:
    """Drop renderer control tokens like <|message_model|> that leak into text."""
    return _CONTROL_TOKEN.sub("", text)


def is_empty_response(text: str | None) -> bool:
    return not text or not strip_control_tokens(text).strip()


def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def _note(extracted: str | None) -> str | None:
    return None if extracted is not None else "unparsed"


def _grade_gsm8k(text: str, gold: str) -> Grade:
    extracted = extract_gsm8k_answer(text) or None
    return Grade(float(check_gsm8k(text, gold)), extracted, _note(extracted))


def _grade_math500(text: str, gold: str) -> Grade:
    try:
        extracted = extract_last_boxed(text) or None
    except ValueError:
        extracted = None
    verdict = grade_answer(extracted, gold) if extracted is not None else False
    return Grade(float(verdict), extracted, _note(extracted))


def _grade_aime(text: str, gold: str) -> Grade:
    # WHY last-boxed: models \boxed intermediate results; the final answer is
    # the last one. extract_boxed() (first match) produced false negatives.
    try:
        boxed = extract_last_boxed(text) or None
    except ValueError:
        boxed = None
    extracted = (extract_number(boxed) if boxed else extract_gsm8k_answer(text)) or None
    try:
        verdict = int(float(extracted)) == int(gold)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        verdict = False
    return Grade(float(verdict), extracted, _note(extracted))


def _grade_exact(text: str, gold: str) -> Grade:
    extracted = _norm(text)
    return Grade(float(extracted == _norm(gold)), extracted, None)


def _grade_number(text: str, gold: str) -> Grade:
    nums = re.findall(r"[-]?\d+[,\d]*\.?\d*", text)
    extracted = extract_number(nums[-1]) if nums else None
    verdict = extracted is not None and _norm(extracted) == _norm(gold)
    return Grade(float(verdict), extracted, _note(extracted))


def _grade_boxed(text: str, gold: str) -> Grade:
    try:
        extracted = extract_last_boxed(text)
    except ValueError:
        extracted = None
    verdict = extracted is not None and _norm(extracted) == _norm(gold)
    return Grade(float(verdict), extracted, _note(extracted))


def _grade_contains(text: str, gold: str) -> Grade:
    found = _norm(gold) in _norm(text)
    return Grade(float(found), gold if found else None, _note(gold if found else None))


GRADERS: dict[str, tuple[GraderFn, str]] = {
    "gsm8k": (_grade_gsm8k, "GSM8K-style extraction (boxed, ####, answer is) then numeric compare"),
    "math500": (_grade_math500, "last \\boxed{} then symbolic equivalence (math_grading)"),
    "aime": (_grade_aime, "last boxed-or-fallback extraction then integer compare"),
    "exact": (_grade_exact, "normalized full text equals gold"),
    "number": (_grade_number, "last numeric literal equals gold (string compare)"),
    "boxed": (_grade_boxed, "last \\boxed{} content equals gold, normalized"),
    "contains": (_grade_contains, "normalized gold appears in the text"),
}

_REGEX_PREFIX = "regex:"

# graders whose job is to pull an answer out of free text — extracted=None
# means "nothing to grade" (parse failure). Predicate/custom graders return
# extracted=None on a legitimate verdict-0, which is "failed", not "parse".
_EXTRACTION = {"gsm8k", "math500", "aime", "number", "boxed"}


def extracts_answer(grader_id: str) -> bool:
    return grader_id in _EXTRACTION or grader_id.startswith(_REGEX_PREFIX)


def _grade_regex(grader_id: str, text: str, gold: str) -> Grade:
    try:
        pat = re.compile(grader_id[len(_REGEX_PREFIX):])
    except re.error as e:
        raise UnknownGraderError(f"{grader_id!r}: bad pattern: {e}") from e
    m = pat.search(text)
    extracted = (m.group(1) if pat.groups else m.group(0)) if m else None
    verdict = extracted is not None and _norm(extracted) == _norm(gold)
    return Grade(float(verdict), extracted, _note(extracted))


def _grade_custom(grader_id: str, text: str, gold: str) -> Grade:
    fn = _import_grader(grader_id)
    res = fn(text, gold)
    if isinstance(res, Grade):
        return res
    if isinstance(res, (int, float)):
        return Grade(float(res), None, None)
    return Grade(0.0, None, "bad return")


def _import_grader(grader_id: str) -> Callable:
    mod_name, _, fn_name = grader_id.partition(":")
    try:
        if mod_name.endswith(".py") or "/" in mod_name:
            # console scripts don't put cwd on sys.path, so a plain file path
            # is the reliable way to point at an uninstalled grader
            spec = importlib.util.spec_from_file_location("flipbook_custom_grader", mod_name)
            if spec is None or spec.loader is None:
                raise ImportError(mod_name)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
    except (ImportError, AttributeError, FileNotFoundError) as e:
        raise UnknownGraderError(_unknown_msg(grader_id)) from e
    if not callable(fn):
        raise UnknownGraderError(f"{grader_id!r}: {fn_name} is not callable")
    return fn


def _unknown_msg(grader_id: str) -> str:
    return (
        f"unknown grader {grader_id!r}; registered: {sorted(GRADERS)}; "
        f"dynamic forms: {_REGEX_PREFIX}<pattern>, module.path:func"
    )


def grade(grader_id: str, text: str, gold: str) -> Grade:
    if grader_id in GRADERS:
        fn = GRADERS[grader_id][0]
    elif grader_id.startswith(_REGEX_PREFIX):
        def fn(t, g):
            return _grade_regex(grader_id, t, g)
    elif ":" in grader_id:
        def fn(t, g):
            return _grade_custom(grader_id, t, g)
    else:
        raise UnknownGraderError(_unknown_msg(grader_id))
    # WHY here, not per grader: an end-of-message-only generation must read as
    # "empty" under every grader, including ones that would extract "".
    if is_empty_response(text):
        return Grade(0.0, None, "empty response")
    return fn(text, gold)


def validate_graders(rows_or_ids: Iterable) -> set[str]:
    """grader_ids in a manifest (rows or bare ids) that grade() cannot resolve."""
    ids = {(r["grader_id"] if isinstance(r, dict) else r) for r in rows_or_ids}
    bad = set()
    for gid in ids:
        if gid in GRADERS:
            continue
        if gid.startswith(_REGEX_PREFIX):
            try:
                re.compile(gid[len(_REGEX_PREFIX):])
            except re.error:
                bad.add(gid)
            continue
        if ":" in gid:
            try:
                _import_grader(gid)
            except Exception:  # noqa: BLE001 — user modules can raise anything at import
                bad.add(gid)
            continue
        bad.add(gid)
    return bad
