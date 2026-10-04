from flipbook.graders import UnknownGraderError, grade, strip_control_tokens, validate_graders


def test_gsm8k_boxed_correct():
    g = grade("gsm8k", "work work \\boxed{42}", "42")
    assert g.verdict == 1.0
    assert g.extracted == "42"
    assert g.note is None


def test_gsm8k_wrong():
    g = grade("gsm8k", "\\boxed{41}", "42")
    assert g.verdict == 0.0
    assert g.extracted == "41"


def test_gsm8k_no_number():
    g = grade("gsm8k", "I give up", "42")
    assert g.verdict == 0.0
    assert g.extracted is None
    assert g.note == "unparsed"


def test_math500_symbolic_equivalence():
    g = grade("math500", "thus \\boxed{\\frac{1}{2}}", "0.5")
    assert g.verdict == 1.0
    assert g.extracted == "\\frac{1}{2}"


def test_math500_unboxed():
    g = grade("math500", "the answer is 0.5", "0.5")
    assert g.verdict == 0.0
    assert g.extracted is None
    assert g.note == "unparsed"


def test_aime_padded_int():
    g = grade("aime", "answer: \\boxed{042}", "42")
    assert g.verdict == 1.0
    assert g.extracted == "042"  # raw extraction; the int compare absorbs padding


def test_aime_unboxed_fallback_still_extracts():
    g = grade("aime", "the answer is 7", "42")
    assert g.verdict == 0.0
    assert g.extracted == "7"
    assert g.note is None


def test_exact_normalizes_whitespace_and_case():
    assert grade("exact", "  The   Answer\nIs 42 ", "the answer is 42").verdict == 1.0
    assert grade("exact", "the answer is 41", "the answer is 42").verdict == 0.0


def test_number_picks_last_literal():
    g = grade("number", "tried 3 then settled on 42", "42")
    assert g.verdict == 1.0 and g.extracted == "42"
    assert grade("number", "the answer is 41", "42").verdict == 0.0
    assert grade("number", "no digits", "42").extracted is None


def test_boxed_and_contains():
    assert grade("boxed", "so \\boxed{\\frac{1}{2}}", "\\frac{1}{2}").verdict == 1.0
    assert grade("boxed", "no box", "x").note == "unparsed"
    assert grade("contains", "the capital is Paris, obviously", "paris").verdict == 1.0
    assert grade("contains", "london", "paris").verdict == 0.0


def test_regex_grader_first_group_then_whole_match():
    g = grade("regex:answer is (\\w+)", "the answer is Blue.", "blue")
    assert g.verdict == 1.0 and g.extracted == "Blue"
    assert grade("regex:\\d+", "there were 7 dwarves", "7").verdict == 1.0
    assert grade("regex:\\d+", "no digits", "7").extracted is None


def test_dotted_import_grader(tmp_path, monkeypatch):
    (tmp_path / "mygraders.py").write_text(
        "from flipbook.graders import Grade\n"
        "def eq(text, gold):\n    return text.strip() == gold\n"
        "def rich(text, gold):\n    return Grade(1.0, 'custom')\n"
        "def weird(text, gold):\n    return 'yes'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    g = grade("mygraders:eq", "  hi ", "hi")
    assert g.verdict == 1.0 and g.extracted is None
    assert grade("mygraders:rich", "a", "b").extracted == "custom"
    assert grade("mygraders:weird", "a", "b").note == "bad return"


def test_aime_uses_last_boxed():
    # models \boxed intermediate results; the final answer is the last one
    g = grade("aime", "so \\boxed{x=4} or \\boxed{y=38}. Sum: \\boxed{50}", "50")
    assert g.verdict == 1.0 and g.extracted == "50"
    assert grade("boxed", "first \\boxed{9} then \\boxed{50}", "50").verdict == 1.0


def test_unknown_grader_raises():
    try:
        grade("nope", "x", "y")
    except ValueError as e:
        assert "unknown grader" in str(e)
        assert "gsm8k" in str(e)  # names the registered ids
        assert "regex:" in str(e) and "module.path:func" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_unknown_dotted_grader_raises():
    try:
        grade("no.such.module:f", "x", "y")
    except UnknownGraderError:
        pass
    else:
        raise AssertionError("expected UnknownGraderError")


def test_file_path_grader(tmp_path):
    f = tmp_path / "mygraders.py"
    f.write_text(
        "def shout(text, gold):\n"
        "    from flipbook.graders import Grade\n"
        "    return Grade(float(text == text.upper()), None, None)\n"
    )
    assert grade(f"{f}:shout", "LOUD", "x").verdict == 1.0
    assert grade(f"{f}:shout", "quiet", "x").verdict == 0.0


def test_validate_graders():
    rows = [
        {"grader_id": "gsm8k"},
        {"grader_id": "regex:\\d+"},
        {"grader_id": "regex:("},
        {"grader_id": "bogus"},
    ]
    assert validate_graders(rows) == {"regex:(", "bogus"}


def test_empty_response_beats_parse():
    g = grade("gsm8k", "<|content_model_end_sampling|>", "42")
    assert g.verdict == 0.0 and g.extracted is None and g.note == "empty response"
    assert grade("aime", "  \n", "42").note == "empty response"
    # exact would otherwise extract "" and compare it
    assert grade("exact", "<|message_model|>", "x").note == "empty response"
    # non-empty but unparseable is still a parse failure
    assert grade("gsm8k", "I give up", "42").note == "unparsed"


def test_strip_control_tokens():
    assert strip_control_tokens("<|message_model|>hi <|x|>there") == "hi there"
    assert strip_control_tokens("a < b | c > d") == "a < b | c > d"
