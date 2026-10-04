from __future__ import annotations

from local_ai_dictation.benchmark import (
    expected_sidecar_path,
    load_expected_transcript,
    load_normalized_expected_transcript,
    normalize_transcript,
    normalized_exact_match,
)


def test_normalize_transcript_applies_nfkc_casefold_punctuation_and_whitespace_rules():
    assert normalize_transcript("  Ｈｅｌｌｏ，\nWORLD!!  ") == "hello world"


def test_normalized_exact_match_ignores_case_and_punctuation_only_differences():
    assert normalized_exact_match("Hello, world!!!", "hello world") is True


def test_normalized_exact_match_rejects_lexical_differences():
    assert normalized_exact_match("hello there", "hello world") is False


def test_expected_sidecar_loader_discovers_and_normalizes_expected_text(tmp_path):
    fixture_path = tmp_path / "short_16k.wav"
    fixture_path.write_bytes(b"RIFF")
    sidecar_path = expected_sidecar_path(fixture_path)
    sidecar_path.write_text(" Example, TRANSCRIPT!\n", encoding="utf-8")

    assert sidecar_path.name == "short_16k.expected.txt"
    assert load_expected_transcript(fixture_path) == " Example, TRANSCRIPT!\n"
    assert load_normalized_expected_transcript(fixture_path) == "example transcript"
