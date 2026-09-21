from datetime import datetime, timedelta, timezone

import pytest

from auditor_stt.captions import (
    Cue,
    format_timestamp,
    join_words,
    make_cues,
    parse_start_time,
    split_lines,
    to_srt,
    to_text,
    to_vtt,
    to_youtube,
)


def _word(text, start, end):
    return {"start": start, "end": end, "text": text}


def _result(*segments):
    return {"segments": list(segments)}


def _segment(words, text=None, start=None, end=None):
    return {
        "start": words[0]["start"] if start is None and words else start,
        "end": words[-1]["end"] if end is None and words else end,
        "text": text if text is not None else " ".join(w["text"] for w in words),
        "words": words,
    }


GREETING = _result(
    _segment(
        [
            _word(" Hyvää", 0.0, 0.5),
            _word(" huomenta.", 0.5, 1.0),
            _word(" Tervetuloa", 1.2, 1.9),
            _word(" kaikille", 1.9, 2.4),
        ]
    )
)


# --- lcyt golden strings ---------------------------------------------------
# Expected values were produced once by running live-captions-yt's own
# YoutubeLiveCaptionSender._format_timestamp / _build_caption_body and joining
# the records the way send_batch does ("\n".join(records) + "\n"); sender.py
# (python-packages/lcyt/lcyt/sender.py) is the source of truth for this format.

GOLDEN_CUES = [
    Cue(0.0, 1.0, "Hyvää huomenta"),
    Cue(1.5, 2.5, "ja tervetuloa."),
    Cue(3671.25, 3672.0, "Tunnin päästä"),
]
GOLDEN_START = datetime(2026, 3, 15, 10, 30, 0, 123456, tzinfo=timezone.utc)

GOLDEN_PLAIN = (
    "2026-03-15T10:30:00.123\nHyvää huomenta\n"
    "2026-03-15T10:30:01.623\nja tervetuloa.\n"
    "2026-03-15T11:31:11.373\nTunnin päästä\n"
)
GOLDEN_REGION = (
    "2026-03-15T10:30:00.123 region:reg1#cue1\nHyvää huomenta\n"
    "2026-03-15T10:30:01.623 region:reg1#cue1\nja tervetuloa.\n"
    "2026-03-15T11:31:11.373 region:reg1#cue1\nTunnin päästä\n"
)
GOLDEN_REGION_CUSTOM = (
    "2026-03-15T10:30:00.123 region:reg2#cue7\nHyvää huomenta\n"
    "2026-03-15T10:30:01.623 region:reg2#cue7\nja tervetuloa.\n"
    "2026-03-15T11:31:11.373 region:reg2#cue7\nTunnin päästä\n"
)


def test_youtube_matches_lcyt_golden_without_region():
    assert to_youtube(GOLDEN_CUES, GOLDEN_START) == GOLDEN_PLAIN


def test_youtube_matches_lcyt_golden_with_region_and_cue():
    assert to_youtube(GOLDEN_CUES, GOLDEN_START, region="reg2", cue="cue7") == GOLDEN_REGION_CUSTOM


def test_youtube_region_alone_uses_lcyt_default_cue():
    assert to_youtube(GOLDEN_CUES, GOLDEN_START, region="reg1") == GOLDEN_REGION
    assert to_youtube(GOLDEN_CUES, GOLDEN_START, cue="cue1") == GOLDEN_REGION


def test_youtube_naive_start_time_is_utc():
    assert to_youtube(GOLDEN_CUES, GOLDEN_START.replace(tzinfo=None)) == GOLDEN_PLAIN


def test_youtube_converts_aware_start_time_to_utc():
    helsinki = timezone(timedelta(hours=2))
    start = GOLDEN_START.astimezone(helsinki)
    assert start.hour == 12
    assert to_youtube(GOLDEN_CUES, start) == GOLDEN_PLAIN


def test_youtube_truncates_to_milliseconds_and_always_prints_them():
    cues = [Cue(0.0, 1.0, "a"), Cue(0.9996, 2.0, "b")]
    start = datetime(2026, 3, 15, 10, 30, 0, 999999, tzinfo=timezone.utc)
    # lcyt's Python sender prints a whole second without a fraction; its JS twin
    # and the YouTube format description always carry .mmm, which we follow.
    assert to_youtube(cues, datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc)) == (
        "2026-03-15T10:30:00.000\na\n2026-03-15T10:30:01.000\nb\n"
    )
    assert to_youtube(cues, start) == "2026-03-15T10:30:00.999\na\n2026-03-15T10:30:01.999\nb\n"


def test_youtube_rolls_over_midnight_and_year():
    start = datetime(2026, 12, 31, 23, 59, 58, 999999, tzinfo=timezone.utc)
    # Same expectation as lcyt's _format_timestamp(start + 2.5 s).
    assert to_youtube([Cue(2.5, 3.0, "x")], start) == "2027-01-01T00:00:01.499\nx\n"


def test_youtube_flattens_multiline_cue_text_to_one_line():
    cues = [Cue(0.0, 2.0, "eka rivi\ntoinen  rivi\r\nkolmas")]
    body = to_youtube(cues, GOLDEN_START)
    assert body == "2026-03-15T10:30:00.123\neka rivi toinen rivi kolmas\n"
    assert len(body.splitlines()) == 2


def test_youtube_skips_blank_cues_and_handles_empty_input():
    assert to_youtube([], GOLDEN_START) == ""
    assert to_youtube([Cue(0.0, 1.0, " \n ")], GOLDEN_START) == ""
    assert to_youtube([Cue(0.0, 1.0, " \n "), Cue(1.0, 2.0, "a")], GOLDEN_START) == (
        "2026-03-15T10:30:01.123\na\n"
    )


def test_youtube_rejects_whitespace_in_region_or_cue():
    with pytest.raises(ValueError):
        to_youtube(GOLDEN_CUES, GOLDEN_START, region="reg 1")
    with pytest.raises(ValueError):
        to_youtube(GOLDEN_CUES, GOLDEN_START, region="reg1", cue="cue1\nfoo")


# --- start_time parsing ----------------------------------------------------


def test_parse_start_time_accepts_z_offsets_and_naive():
    expected = datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
    assert parse_start_time("2026-03-15T10:30:00Z") == expected
    assert parse_start_time("2026-03-15T12:30:00+02:00") == expected
    assert parse_start_time("2026-03-15T10:30:00").tzinfo == timezone.utc
    assert parse_start_time("2026-03-15T10:30:00.250Z").microsecond == 250000
    with pytest.raises(ValueError):
        parse_start_time("yesterday")


# --- timestamps and text helpers -------------------------------------------


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3661.5) == "01:01:01.500"
    assert format_timestamp(3661.5, ",") == "01:01:01,500"
    assert format_timestamp(-2) == "00:00:00.000"
    assert format_timestamp(100 * 3600) == "100:00:00.000"


def test_join_words_attaches_punctuation():
    words = [_word(" (hei", 0, 1), _word("maailma", 1, 2), _word(")", 2, 3), _word(".", 3, 4)]
    assert join_words(words) == "(hei maailma)."


def test_split_lines_wraps_to_two_lines():
    assert split_lines("a" * 30 + " " + "b" * 30, 42) == "a" * 30 + "\n" + "b" * 30
    assert split_lines("lyhyt teksti", 42) == "lyhyt teksti"
    assert split_lines("   ", 42) == ""


# --- cue grouping ----------------------------------------------------------


def test_make_cues_breaks_on_sentence_punctuation():
    assert make_cues(GREETING) == [
        Cue(0.0, 1.0, "Hyvää huomenta."),
        Cue(1.2, 2.4, "Tervetuloa kaikille"),
    ]


def test_make_cues_splits_by_max_duration():
    words = [_word(f"w{i}", float(i), float(i + 1)) for i in range(10)]
    cues = make_cues(_result(_segment(words)), max_duration=3.0)
    assert [(c.start, c.end) for c in cues] == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 10.0)]
    assert cues[0].text == "w0 w1 w2"


def test_make_cues_splits_by_max_chars_and_wraps_lines():
    words = [_word("abcdefghij", float(i) * 0.1, float(i) * 0.1 + 0.1) for i in range(3)]
    cues = make_cues(_result(_segment(words)), max_chars=10)
    # Two ten-letter words hit the 2 * max_chars limit; the third starts a new cue.
    assert [c.text for c in cues] == ["abcdefghij\nabcdefghij", "abcdefghij"]


def test_make_cues_groups_words_across_consecutive_segments():
    first = _segment([_word("Hyvää", 0.0, 0.5)])
    second = _segment([_word("huomenta", 0.6, 1.1)])
    assert make_cues(_result(first, second)) == [Cue(0.0, 1.1, "Hyvää huomenta")]


def test_make_cues_falls_back_to_segment_without_words():
    with_words = _segment([_word("Alku", 0.0, 1.0)])
    no_words = {"start": 2.0, "end": 4.5, "text": " " + "a" * 30 + " " + "b" * 30 + " ", "words": []}
    missing_words_key = {"start": 5.0, "end": 6.0, "text": "Loppu"}
    cues = make_cues(_result(with_words, no_words, missing_words_key))
    assert cues == [
        Cue(0.0, 1.0, "Alku"),
        Cue(2.0, 4.5, "a" * 30 + "\n" + "b" * 30),
        Cue(5.0, 6.0, "Loppu"),
    ]


def test_make_cues_word_less_segment_ends_the_running_cue():
    first = _segment([_word("Ennen", 0.0, 0.5)])
    plain = {"start": 1.0, "end": 2.0, "text": "Ilman sanoja", "words": []}
    last = _segment([_word("Jälkeen", 3.0, 3.5)])
    cues = make_cues(_result(first, plain, last))
    assert [c.text for c in cues] == ["Ennen", "Ilman sanoja", "Jälkeen"]


def test_make_cues_uses_segment_when_no_word_has_timestamps():
    segment = {"start": 1.0, "end": 2.0, "text": "Ei aikoja", "words": [{"text": "Ei"}, {"text": "aikoja"}]}
    assert make_cues(_result(segment)) == [Cue(1.0, 2.0, "Ei aikoja")]


def test_make_cues_skips_untimed_words_in_a_timed_segment():
    words = [_word("Yksi", 0.0, 0.5), {"text": "ei"}, _word("kaksi", 0.5, 1.0)]
    assert make_cues(_result(_segment(words))) == [Cue(0.0, 1.0, "Yksi kaksi")]


def test_make_cues_empty_input():
    assert make_cues({}) == []
    assert make_cues({"segments": []}) == []
    assert make_cues({"segments": [{"start": 0, "end": 1, "text": "  ", "words": []}]}) == []


# --- output formats --------------------------------------------------------


def test_to_vtt_shape():
    cues = make_cues(GREETING)
    assert to_vtt(cues) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\nHyvää huomenta.\n\n"
        "00:00:01.200 --> 00:00:02.400\nTervetuloa kaikille\n\n"
    )


def test_to_vtt_escapes_markup_but_not_quotes():
    assert to_vtt([Cue(0, 1, 'Tom & <Jerry> "hei"')]) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTom &amp; &lt;Jerry&gt; \"hei\"\n\n"
    )


def test_to_srt_shape():
    cues = make_cues(GREETING)
    assert to_srt(cues) == (
        "1\n00:00:00,000 --> 00:00:01,000\nHyvää huomenta.\n\n"
        "2\n00:00:01,200 --> 00:00:02,400\nTervetuloa kaikille\n\n"
    )


def test_to_srt_keeps_wrapped_lines_and_does_not_escape():
    assert to_srt([Cue(0, 1, "a & b\nc")]) == "1\n00:00:00,000 --> 00:00:01,000\na & b\nc\n\n"


def test_empty_outputs():
    assert to_vtt([]) == "WEBVTT\n\n"
    assert to_srt([]) == ""


def test_to_text_prefers_result_text_then_joins_segments():
    assert to_text({"text": "  Hyvää   huomenta. "}) == "Hyvää huomenta."
    assert to_text(GREETING) == "Hyvää huomenta. Tervetuloa kaikille"
    assert to_text({"segments": []}) == ""
    assert to_text({}) == ""
