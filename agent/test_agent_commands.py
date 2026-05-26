"""Unit tests for agent self-management command parsing and stream filtering.

Pure logic - no GUI or LLM needed. Run: python test_agent_commands.py
"""

from agent_commands import CommandStreamFilter, parse_commands


def _stream(text, chunk_size):
    """Feed text through the filter in fixed-size chunks; return visible output."""
    f = CommandStreamFilter()
    out = []
    for i in range(0, len(text), chunk_size):
        out.append(f.feed(text[i:i + chunk_size]))
    out.append(f.flush())
    return "".join(out)


def test_parse_extracts_all_command_types():
    raw = (
        "Sure thing.[[LOCATION: grocery store]] I'll add that. "
        "[[NOTE: pick up milk]] Also noting your taste.[[REMEMBER: prefers dark roast]] "
        "Done with the dentist.[[NOTE_DONE: call dentist]]"
    )
    clean, cmds = parse_commands(raw)
    assert ("LOCATION", "grocery store") in cmds
    assert ("NOTE", "pick up milk") in cmds
    assert ("REMEMBER", "prefers dark roast") in cmds
    assert ("NOTE_DONE", "call dentist") in cmds
    assert "[[" not in clean and "]]" not in clean
    assert "grocery store" not in clean  # command value should not leak
    assert "Sure thing." in clean and "I'll add that." in clean


def test_parse_handles_no_commands():
    clean, cmds = parse_commands("Just a normal reply with no commands.")
    assert cmds == []
    assert clean == "Just a normal reply with no commands."


def test_parse_is_case_insensitive_and_trims():
    clean, cmds = parse_commands("ok [[location:   the office   ]] done")
    assert cmds == [("LOCATION", "the office")]
    assert "office" not in clean


def test_non_command_brackets_pass_through():
    clean, cmds = parse_commands("See note [[1]] and array a[[i]] here.")
    assert cmds == []
    assert "[[1]]" in clean and "a[[i]]" in clean


def test_stream_hides_commands_across_chunk_sizes():
    raw = (
        "Heading out now.[[LOCATION: the gym]] Have a good workout! "
        "[[NOTE: bring water bottle]] See you later."
    )
    expected_visible = "Heading out now. Have a good workout!  See you later."
    for size in range(1, len(raw) + 1):
        visible = _stream(raw, size)
        assert "[[" not in visible, f"command leaked at chunk size {size}: {visible!r}"
        assert "the gym" not in visible, f"value leaked at chunk size {size}"
        assert "bring water bottle" not in visible, f"value leaked at chunk size {size}"
        assert "Heading out now." in visible and "See you later." in visible


def test_stream_passes_non_command_brackets():
    raw = "Refer to [[footnote]] and the [[2]] item."
    for size in range(1, len(raw) + 1):
        visible = _stream(raw, size)
        assert "[[footnote]]" in visible, f"chunk size {size}: {visible!r}"
        assert "[[2]]" in visible


def test_stream_handles_unclosed_command_at_end():
    # Model got cut off mid-command; the partial text should still surface.
    raw = "All set. [[NOTE: finish the rep"
    visible = _stream(raw, 3)
    assert "All set." in visible
    assert "[[NOTE: finish the rep" in visible  # unclosed -> shown, not swallowed
    # And parse_commands should not treat it as a completed command.
    _, cmds = parse_commands(raw)
    assert cmds == []


def test_stream_matches_parse_modulo_whitespace():
    raw = "abc[[NOTE: x]]def[[LOCATION: home]]ghi"
    visible = "".join(_stream(raw, 2).split())
    clean, _ = parse_commands(raw)
    assert visible == "".join(clean.split())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
