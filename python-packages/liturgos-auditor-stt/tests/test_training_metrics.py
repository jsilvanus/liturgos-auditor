import random

import pytest

from auditor_stt.training.metrics import (
    cer,
    edit_distance,
    metrics_for,
    normalize_for_wer,
    per_speaker,
    wer,
)


def _naive_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


# --- edit_distance -----------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("kitten", "sitting", 3),
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("kärry", "karry", 1),  # one substitution, not two: ä is one character
        ("saarna", "saarnaa", 1),
        ("abcdef", "azced", 3),
    ],
)
def test_edit_distance_known_values(a, b, expected):
    assert edit_distance(a, b) == expected
    assert edit_distance(b, a) == expected  # symmetric


def test_edit_distance_works_on_word_lists():
    assert edit_distance("hyvää huomenta kaikille".split(), "hyvää iltaa".split()) == 2
    assert edit_distance([], ["a"]) == 1


def test_edit_distance_matches_a_plain_dynamic_programme_on_random_input():
    rng = random.Random(7)
    for _ in range(1500):
        alphabet = "abcdefgäö"[: rng.choice([2, 3, 5, 9])]
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert edit_distance(a, b) == _naive_distance(a, b), (a, b)
        assert edit_distance(list(a), list(b)) == _naive_distance(a, b)


def test_edit_distance_handles_a_long_sermon_sized_input_quickly():
    rng = random.Random(3)
    words = [f"sana{rng.randint(0, 2000)}" for _ in range(8000)]
    hypothesis = [w if rng.random() > 0.2 else "x" for w in words]
    expected = sum(1 for w, h in zip(words, hypothesis) if w != h)
    # only substitutions were made, so the distance cannot exceed that count
    assert 0 < edit_distance(words, hypothesis) <= expected


# --- normalize_for_wer ---------------------------------------------------------------


def test_normalisation_lowercases_and_drops_punctuation():
    assert normalize_for_wer("Hyvää päivää, Ärjä!") == "hyvää päivää ärjä"
    assert normalize_for_wer("«Kyllä», sanoi hän: \"ehkä\"...") == "kyllä sanoi hän ehkä"


def test_normalisation_turns_hyphens_and_dashes_into_word_breaks():
    assert normalize_for_wer("kirkko-orkesteri") == "kirkko orkesteri"
    assert normalize_for_wer("sana – toinen—kolmas") == "sana toinen kolmas"
    assert normalize_for_wer("EU:n") == "eu n"


def test_normalisation_keeps_digits_and_drops_symbols():
    assert normalize_for_wer("Psalmi 23: 1-6 (50 %)") == "psalmi 23 1 6 50"
    assert normalize_for_wer("5 € + 3") == "5 3"


def test_normalisation_composes_unicode_and_collapses_whitespace():
    assert normalize_for_wer("Äiti") == "äiti"  # A + combining diaeresis, as some tools write it
    assert normalize_for_wer("ä") == normalize_for_wer("ä") == "ä"
    assert normalize_for_wer("  yksi \n\t kaksi   ") == "yksi kaksi"


def test_normalisation_of_nothing_is_empty():
    assert normalize_for_wer(None) == ""
    assert normalize_for_wer("") == ""
    assert normalize_for_wer(" ... !? ") == ""


# --- wer / cer -----------------------------------------------------------------------


def test_wer_counts_substitutions_insertions_and_deletions():
    assert wer(["a b c"], ["a b c"]) == 0.0
    assert wer(["a b c"], ["a x c d"]) == pytest.approx(2 / 3, abs=1e-6)  # one substitution, one insertion
    assert wer(["a b c d"], ["a d"]) == 0.5  # two deletions
    assert wer(["a b c"], [""]) == 1.0
    assert wer(["a b"], ["c d e f"]) == 2.0  # WER can exceed 1


def test_wer_is_corpus_level_not_a_mean_of_utterance_rates():
    # 0/2 words wrong in the first, 3/4 in the second: corpus 3/6, a per-utterance mean would give 0.375
    assert wer(["a b", "c d e f"], ["a b", "c"]) == 0.5


def test_wer_compares_the_words_as_given():
    assert wer(["Hyvää päivää"], ["hyvää päivää"]) == 0.5
    assert wer(["päivää,"], ["päivää"]) == 1.0


def test_an_empty_reference_adds_its_insertions_but_no_words():
    assert wer(["a b", ""], ["a b", "extra words"]) == 1.0  # 2 insertions over 2 reference words
    assert wer(["a b", ""], ["a b", ""]) == 0.0


def test_wer_is_undefined_without_any_reference_words():
    assert wer([], []) is None
    assert wer([""], [""]) is None
    assert wer(["", "  "], ["something", ""]) is None
    assert cer([""], ["x"]) is None


def test_wer_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="2 references but 1 hypotheses"):
        wer(["a", "b"], ["a"])


def test_cer_counts_characters_including_umlauts_and_spaces():
    assert cer(["kissa"], ["kisa"]) == pytest.approx(0.2)
    assert cer(["päivä"], ["paiva"]) == pytest.approx(0.4)  # ä and ä are two substitutions
    assert cer(["ab cd"], ["abcd"]) == pytest.approx(0.2)  # the space is a character
    assert cer(["ab  cd \n"], [" ab cd"]) == 0.0  # whitespace runs and the ends do not count


def test_rates_are_rounded_to_six_digits():
    assert wer(["a b c"], ["a b"]) == 0.333333


# --- metrics_for ---------------------------------------------------------------------


def test_metrics_for_reports_raw_and_normalised_side_by_side():
    result = metrics_for([("Hyvää päivää, Ärjä.", "hyvää päivää ärjä")])

    assert result["n"] == 1
    assert result["reference_words"] == 3
    assert result["wer_normalised"] == 0.0 and result["cer_normalised"] == 0.0
    assert result["wer_raw"] == 1.0  # every word differs in case or punctuation
    assert 0 < result["cer_raw"] < 1


def test_metrics_for_pools_over_pairs_and_treats_none_as_empty():
    result = metrics_for([("yksi kaksi", "yksi kaksi"), ("kolme neljä", None), (None, "")])

    assert result["n"] == 3
    assert result["reference_words"] == 4
    assert result["wer_normalised"] == 0.5  # the None hypothesis deletes both words
    assert metrics_for([])["wer_normalised"] is None and metrics_for([])["n"] == 0


# --- per_speaker ---------------------------------------------------------------------


def _record(speaker, reference, hypothesis, seconds):
    return {"speaker_id": speaker, "reference": reference, "hypothesis": hypothesis, "audio_seconds": seconds}


def _records():
    return [
        _record("a", "yksi kaksi kolme neljä", "yksi kaksi kolme neljä", 40.0),  # WER 0
        _record("b", "yksi kaksi kolme neljä", "yksi kaksi kolme viisi", 40.0),  # 0.25
        _record("c", "yksi kaksi kolme neljä", "yksi kaksi x y", 40.0),  # 0.5
        _record("d", "yksi kaksi kolme neljä", "x y z w", 5.0),  # 1.0, but only 5 s of audio
    ]


def test_per_speaker_wer_and_spread():
    result = per_speaker(_records())

    assert {s: v["wer"] for s, v in result["speakers"].items()} == {"a": 0.0, "b": 0.25, "c": 0.5, "d": 1.0}
    assert result["n_speakers"] == 4
    # sorted 0, .25, .5, 1: median .375, p90 interpolated 0.5 + 0.7 * 0.5
    assert result["wer"] == {"n": 4, "min": 0.0, "median": 0.375, "p90": 0.85, "max": 1.0}


def test_per_speaker_flags_speakers_with_too_little_audio_and_spreads_the_rest_separately():
    result = per_speaker(_records(), min_audio_seconds=30.0)

    assert result["speakers"]["d"]["low_audio"] is True
    assert result["speakers"]["a"]["low_audio"] is False
    assert result["n_low_audio"] == 1 and result["min_audio_seconds"] == 30.0
    assert result["wer_reliable"] == {"n": 3, "min": 0.0, "median": 0.25, "p90": 0.45, "max": 0.5}


def test_per_speaker_reliable_spread_is_none_when_nobody_has_enough_audio():
    result = per_speaker(_records(), min_audio_seconds=1000.0)
    assert result["n_low_audio"] == 4
    assert result["wer_reliable"] is None
    assert result["wer"]["n"] == 4


def test_one_bad_speaker_shows_in_the_max_even_when_the_pooled_wer_looks_fine():
    records = [_record(f"s{i}", " ".join(["sana"] * 50), " ".join(["sana"] * 50), 60.0) for i in range(9)]
    records.append(_record("bad", " ".join(["sana"] * 50), " ".join(["muu"] * 25 + ["sana"] * 25), 60.0))

    pooled = metrics_for((r["reference"], r["hypothesis"]) for r in records)["wer_normalised"]
    result = per_speaker(records)

    assert pooled == 0.05
    assert result["wer"]["max"] == 0.5 and result["wer"]["median"] == 0.0


def test_per_speaker_pools_a_speakers_utterances_at_corpus_level():
    result = per_speaker([
        _record("a", "yksi kaksi", "yksi kaksi", 20.0),
        _record("a", "kolme neljä viisi kuusi", "kolme", 20.0),
    ])
    speaker = result["speakers"]["a"]
    assert speaker["wer"] == 0.5  # 3 of 6 words, not the mean of 0 and 0.75
    assert speaker["utterances"] == 2 and speaker["reference_words"] == 6 and speaker["audio_seconds"] == 40.0
    assert speaker["low_audio"] is False


def test_per_speaker_counts_records_without_a_speaker_id_but_does_not_score_them():
    result = per_speaker([_record(None, "a b", "x y", 5.0), _record("", "a b", "x y", 5.0), *_records()])
    assert result["unattributed_utterances"] == 2
    assert result["n_speakers"] == 4


def test_per_speaker_leaves_speakers_without_reference_words_out_of_the_spread():
    result = per_speaker([_record("a", "", "noise", 60.0), _record("b", "yksi kaksi", "yksi kaksi", 60.0)])
    assert result["speakers"]["a"]["wer"] is None
    assert result["n_speakers"] == 2
    assert result["wer"] == {"n": 1, "min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}


def test_per_speaker_without_any_speakers():
    result = per_speaker([_record(None, "a", "a", 1.0)])
    assert result["n_speakers"] == 0 and result["wer"] is None and result["wer_reliable"] is None
    assert result["speakers"] == {}
