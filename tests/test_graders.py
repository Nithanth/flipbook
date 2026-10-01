from flipbook.graders import grade


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


def test_unknown_grader_raises():
    try:
        grade("nope", "x", "y")
    except ValueError as e:
        assert "unknown grader" in str(e)
    else:
        raise AssertionError("expected ValueError")
