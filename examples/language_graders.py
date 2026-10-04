"""Custom graders for the language_tasks example manifest.

Referenced from examples/language_tasks.jsonl as `examples.language_graders:FN`
(the module.path:func escape hatch — run `flipbook eval` from the repo root so
the module imports). Each takes (text, gold) and returns a Grade or bool.

`judge` is a real LLM-as-judge: it samples Inkling-Small through Tinker to
check the response against the `gold` criterion. It needs TINKER_API_KEY in
the environment, and runs its event loop in a worker thread because grade()
is called inside the eval loop.
"""

import json
import re
import threading

from flipbook.graders import Grade

_JUDGE_MODEL = "thinkingmachines/Inkling-Small"
_JUDGE_PROMPT = (
    "You are a strict grader. A model was asked to satisfy a criterion.\n\n"
    "Criterion: {criterion}\n\nModel response:\n{response}\n\n"
    "Does the response satisfy the criterion? Answer with YES or NO only."
)


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


def exactly_n_words(text: str, gold: str) -> Grade:
    n = len(_words(text))
    return Grade(float(n == int(gold)), str(n), f"{n} words")


def at_most_n_words(text: str, gold: str) -> Grade:
    n = len(_words(text))
    return Grade(float(n <= int(gold)), str(n), f"{n} words")


def exactly_n_lines(text: str, gold: str) -> Grade:
    n = len([ln for ln in text.strip().splitlines() if ln.strip()])
    return Grade(float(n == int(gold)), str(n), f"{n} lines")


def no_letter(text: str, gold: str) -> Grade:
    letter = gold.strip().lower()
    ok = letter not in text.lower()
    return Grade(float(ok), None, f"letter {letter!r} {'absent' if ok else 'present'}")


def all_lowercase(text: str, gold: str) -> Grade:
    ok = text.strip() == text.strip().lower()
    return Grade(float(ok), None, "all lowercase" if ok else "has uppercase")


def starts_with(text: str, gold: str) -> Grade:
    got = text.strip()[: len(gold) + 8]
    ok = text.strip().startswith(gold)
    return Grade(float(ok), got[:20], f"must start with {gold!r}")


def json_keys(text: str, gold: str) -> Grade:
    """gold = comma-separated keys the response's JSON object must contain."""
    try:
        obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except (ValueError, IndexError):
        return Grade(0.0, None, "no JSON object found")
    want = {k.strip() for k in gold.split(",") if k.strip()}
    missing = want - set(obj if isinstance(obj, dict) else {})
    return Grade(float(not missing), None, f"missing {sorted(missing)}" if missing else "all keys present")


_judge_state: dict = {}


def _judge_call(criterion: str, response: str) -> bool:
    import asyncio

    async def go() -> str:
        import tinker
        from tinker.types import ModelInput, SamplingParams
        from tinker_cookbook.renderers import get_renderer, get_text_content
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        if "client" not in _judge_state:
            sc = tinker.ServiceClient()
            _judge_state["client"] = sc.create_sampling_client(base_model=_JUDGE_MODEL)
            _judge_state["renderer"] = get_renderer("tml_v0", get_tokenizer(_JUDGE_MODEL))
        client, renderer = _judge_state["client"], _judge_state["renderer"]
        prompt = renderer.build_generation_prompt(
            [{"role": "user", "content": _JUDGE_PROMPT.format(criterion=criterion, response=response)}],
            effort=0.5,
        ).to_ints()
        resp = await client.sample_async(
            ModelInput.from_ints(prompt),
            num_samples=1,
            sampling_params=SamplingParams(
                max_tokens=16, temperature=0.0, stop=renderer.get_stop_sequences()
            ),
        )
        msg, _ = renderer.parse_response(resp.sequences[0].tokens)
        return get_text_content(msg).strip()

    result: list[str] = []
    # grade() runs inside the eval's event loop — a nested asyncio.run would
    # fail, so the judge gets its own loop in a worker thread
    t = threading.Thread(target=lambda: result.append(asyncio.run(go())))
    t.start()
    t.join(timeout=60)
    return result[0].upper().startswith("YES") if result else False


def judge(text: str, gold: str) -> Grade:
    """gold = a natural-language criterion checked by the judge model."""
    try:
        ok = _judge_call(gold, text)
    except Exception as e:  # noqa: BLE001 — judge failures are parse failures, not verdicts
        return Grade(0.0, None, f"judge error: {e}")
    return Grade(float(ok), None, f"judge: {'satisfied' if ok else 'not satisfied'}")
