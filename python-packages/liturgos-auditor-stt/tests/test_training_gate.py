import json
import re
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from datasets import Audio, Dataset, DatasetDict, Features, Value

from auditor_stt import cli
from auditor_stt.serve.model import ModelLoadError
from auditor_stt.training import evaluate, gate

SECRET_REFERENCE = "Herra on minun paimeneni"
SECRET_HYPOTHESIS = "herra on minun paimenen"
RATE = 16000


def _scored(wer, *, version="v-1", per_speaker=None):
    """What evaluate_model returns."""
    return {
        "split": "test",
        "dataset_version": version,
        "audio_seconds": 240.0,
        "n": 40,
        "reference_words": 300,
        "wer_normalised": wer,
        "cer_normalised": None if wer is None else round(wer / 3, 6),
        "wer_raw": None if wer is None else round(wer * 1.5, 6),
        "cer_raw": None if wer is None else round(wer / 2, 6),
        "per_speaker": per_speaker,
    }


def _spread(low, high):
    return {"n": 5, "min": low, "median": (low + high) / 2, "p90": high, "max": high}


def _sanity(**changes):
    return {
        "n_segments": 100, "order_violations": 0, "long_segment_fraction": 0.0,
        "long_segment_seconds": 25.0, "coverage": 0.99, "segments_per_minute": 9.5, **changes,
    }


def _longform(*, regression=False, reasons=(), candidate_wer=0.2, baseline_wer=0.2):
    return {
        "tolerance": 0.02, "chunk_seconds": 60.0, "audio_seconds": 1800.0, "reference_words": 4000,
        "candidate": {"wer": candidate_wer, "words": 3900, "sanity": _sanity()},
        "baseline": {"wer": baseline_wer, "words": 3900, "sanity": _sanity()},
        "wer_delta": round(candidate_wer - baseline_wer, 6),
        "regression": regression, "reasons": list(reasons),
    }


@pytest.fixture
def scoring(monkeypatch, tmp_path):
    """Stub evaluate_model / longform_check; `state` holds the WERs to return and records every call."""
    candidate_dir = tmp_path / "cand"
    candidate_dir.mkdir()
    state = types.SimpleNamespace(
        candidate_dir=candidate_dir,
        wers={"cand": 0.10, "large-v3-turbo": 0.20},
        per_speaker=None,
        version="v-1",
        calls=[],
        longform=_longform(),
        longform_calls=[],
    )

    def fake_evaluate_model(model, dataset_dir, split="test", *, backend="ct2", device="auto", compute_type=None,
                            language="fi", predictions=None):
        state.calls.append({"model": str(model), "backend": backend, "split": split, "device": device,
                            "compute_type": compute_type, "language": language, "collecting": predictions is not None})
        if predictions is not None:
            predictions.append({"recording_id": 1, "speaker_id": "spk", "reference": SECRET_REFERENCE,
                                "hypothesis": SECRET_HYPOTHESIS})
        return _scored(state.wers[Path(str(model)).name], version=state.version, per_speaker=state.per_speaker)

    def fake_longform_check(candidate, baseline, audio, reference, *, tolerance, chunk_seconds, workdir, **options):
        state.longform_calls.append({
            "candidate": str(candidate), "baseline": str(baseline), "tolerance": tolerance,
            "chunk_seconds": chunk_seconds, "workdir": Path(workdir), "workdir_existed": Path(workdir).is_dir(),
            "options": options,
        })
        return state.longform

    monkeypatch.setattr(gate, "evaluate_model", fake_evaluate_model)
    monkeypatch.setattr(gate, "longform_check", fake_longform_check)

    audio = tmp_path / "sermon.wav"
    audio.write_bytes(b"RIFF")
    reference = tmp_path / "sermon.txt"
    reference.write_text("Herra on minun paimeneni.", encoding="utf-8")
    state.audio, state.reference = audio, reference
    return state


def _run(scoring, **options):
    return gate.evaluate_gate(scoring.candidate_dir, "dataset-dir", **options)


def _check(result, name):
    (found,) = [c for c in result["checks"] if c["name"] == name]
    return found


# --- schema --------------------------------------------------------------------------


def test_the_gate_has_exactly_the_documented_schema(scoring):
    result = _run(scoring)

    assert list(result) == [
        "schema", "passed", "created_at", "candidate", "baseline", "dataset_version", "split", "checks", "metrics", "longform",
    ]
    assert result["schema"] == 1 and result["passed"] is True
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", result["created_at"])
    assert result["candidate"] == str(scoring.candidate_dir.resolve())
    assert result["baseline"] == "large-v3-turbo"
    assert result["dataset_version"] == "v-1" and result["split"] == "test"
    assert [c["name"] for c in result["checks"]] == ["beats_baseline_on_test", "longform_no_regression"]
    assert all(set(c) == {"name", "passed", "detail"} and isinstance(c["detail"], str) for c in result["checks"])
    assert list(result["metrics"]) == ["candidate", "baseline", "relative_wer_improvement"]
    assert result["longform"] is None


def test_the_gate_is_written_next_to_the_candidate_and_matches_the_return_value(scoring):
    result = _run(scoring)

    written = json.loads((scoring.candidate_dir / "gate.json").read_text(encoding="utf-8"))
    assert written == result
    assert not (scoring.candidate_dir / "gate.json.tmp").exists()


def test_a_failed_gate_is_still_written(scoring):
    scoring.wers["cand"] = 0.30
    result = _run(scoring)

    assert result["passed"] is False
    assert json.loads((scoring.candidate_dir / "gate.json").read_text(encoding="utf-8"))["passed"] is False


def test_out_overrides_the_location_and_creates_missing_directories(scoring, tmp_path):
    target = tmp_path / "gates" / "run1" / "gate.json"

    _run(scoring, out=target)

    assert target.is_file() and not (scoring.candidate_dir / "gate.json").exists()


def test_a_candidate_that_is_not_a_directory_needs_an_explicit_out(scoring, tmp_path):
    with pytest.raises(ValueError, match="explicit output path"):
        gate.evaluate_gate("large-v3-turbo", "dataset-dir")
    assert scoring.calls == []  # refused before any model was scored

    result = gate.evaluate_gate("cand", "dataset-dir", out=tmp_path / "g.json")
    assert result["candidate"] == "cand"  # not a path on disk: recorded as given
    assert (tmp_path / "g.json").is_file()


def test_metrics_carry_both_models_and_no_bookkeeping_or_ids(scoring):
    scoring.per_speaker = {"n_speakers": 5, "n_low_audio": 1, "unattributed_utterances": 0, "min_audio_seconds": 30.0,
                           "wer": _spread(0.05, 0.4), "wer_reliable": _spread(0.05, 0.2)}

    result = _run(scoring)

    candidate, baseline = result["metrics"]["candidate"], result["metrics"]["baseline"]
    assert candidate["wer_normalised"] == 0.10 and baseline["wer_normalised"] == 0.20
    assert {"n", "audio_seconds", "reference_words", "wer_raw", "cer_raw", "cer_normalised", "per_speaker"} <= set(candidate)
    assert "split" not in candidate and "dataset_version" not in candidate
    assert candidate["per_speaker"]["wer"]["max"] == 0.4  # the spread is in the gate
    assert "speakers" not in candidate["per_speaker"]
    assert result["metrics"]["relative_wer_improvement"] == 0.5


# --- beats_baseline_on_test ------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate, baseline, min_improvement, passed",
    [
        (0.10, 0.20, 0.0, True),
        (0.19, 0.20, 0.0, True),
        (0.20, 0.20, 0.0, False),  # equal is not better
        (0.25, 0.20, 0.0, False),
        (0.19, 0.20, 0.1, False),  # must be below 0.18
        (0.17, 0.20, 0.1, True),
        (0.18, 0.20, 0.1, False),
    ],
)
def test_beats_baseline_is_strict_and_honours_min_improvement(scoring, candidate, baseline, min_improvement, passed):
    scoring.wers.update({"cand": candidate, "large-v3-turbo": baseline})

    result = _run(scoring, min_improvement=min_improvement)

    check = _check(result, "beats_baseline_on_test")
    assert check["passed"] is passed
    assert result["passed"] is passed
    assert f"{candidate:.4f}" in check["detail"] and f"{baseline:.4f}" in check["detail"]


def test_relative_improvement_is_a_fraction_of_the_baseline_wer(scoring):
    scoring.wers.update({"cand": 0.15, "large-v3-turbo": 0.20})
    assert _run(scoring)["metrics"]["relative_wer_improvement"] == 0.25

    scoring.wers.update({"cand": 0.25})
    assert _run(scoring)["metrics"]["relative_wer_improvement"] == -0.25  # worse is negative


def test_an_undefined_wer_fails_the_gate_instead_of_passing_it(scoring):
    scoring.wers["cand"] = None
    result = _run(scoring)
    assert _check(result, "beats_baseline_on_test")["passed"] is False
    assert "undefined" in _check(result, "beats_baseline_on_test")["detail"]
    assert result["passed"] is False and result["metrics"]["relative_wer_improvement"] is None


def test_a_perfect_baseline_cannot_be_beaten_and_has_no_relative_improvement(scoring):
    scoring.wers.update({"cand": 0.0, "large-v3-turbo": 0.0})
    result = _run(scoring)
    assert result["passed"] is False
    assert result["metrics"]["relative_wer_improvement"] is None


def test_split_and_backend_options_reach_the_models(scoring):
    _run(scoring, split="dev", device="cpu", compute_type="int8", language="sv")

    candidate, baseline = scoring.calls
    assert candidate["model"] == str(scoring.candidate_dir) and baseline["model"] == "large-v3-turbo"
    for call in (candidate, baseline):
        assert (call["split"], call["device"], call["compute_type"], call["language"]) == ("dev", "cpu", "int8", "sv")


def test_the_baseline_always_runs_through_ct2_even_when_the_candidate_is_a_hf_checkpoint(scoring):
    _run(scoring, backend="hf")
    assert [c["backend"] for c in scoring.calls] == ["hf", "ct2"]


def test_a_dataset_without_a_version_records_null(scoring):
    scoring.version = None
    assert _run(scoring)["dataset_version"] is None


def test_min_improvement_must_be_a_fraction(scoring):
    for bad in (-0.1, 1.0, 2):
        with pytest.raises(ValueError, match="min_improvement"):
            _run(scoring, min_improvement=bad)
    assert scoring.calls == []


# --- longform_no_regression ------------------------------------------------------------


def test_longform_is_skipped_when_no_material_is_given(scoring):
    result = _run(scoring)

    check = _check(result, "longform_no_regression")
    assert check["passed"] is None and "skipped" in check["detail"]
    assert result["passed"] is True  # skipped checks do not count against the gate
    assert scoring.longform_calls == []


def test_require_longform_turns_the_skip_into_a_failure(scoring):
    result = _run(scoring, require_longform=True)

    assert _check(result, "longform_no_regression")["passed"] is False
    assert result["passed"] is False
    assert _check(result, "beats_baseline_on_test")["passed"] is True


def test_longform_without_regression_passes(scoring):
    result = _run(scoring, longform_audio=scoring.audio, longform_reference=scoring.reference, require_longform=True)

    check = _check(result, "longform_no_regression")
    assert check["passed"] is True and "0.2000" in check["detail"]
    assert result["passed"] is True
    assert result["longform"] == scoring.longform


def test_a_longform_regression_fails_the_gate_even_when_the_test_split_is_won(scoring):
    scoring.longform = _longform(regression=True, reasons=["WER 0.2500 is 0.0500 above the baseline's 0.2000 (tolerance 0.0200)"],
                                 candidate_wer=0.25)

    result = _run(scoring, longform_audio=scoring.audio, longform_reference=scoring.reference)

    assert _check(result, "beats_baseline_on_test")["passed"] is True
    check = _check(result, "longform_no_regression")
    assert check["passed"] is False and "0.0500 above" in check["detail"]
    assert result["passed"] is False
    assert result["longform"]["regression"] is True


def test_longform_runs_in_a_temporary_workdir_that_is_removed_afterwards(scoring):
    _run(scoring, longform_audio=scoring.audio, longform_reference=scoring.reference,
         longform_tolerance=0.05, longform_chunk_seconds=30.0, device="cpu", language="sv")

    (call,) = scoring.longform_calls
    assert call["candidate"] == str(scoring.candidate_dir) and call["baseline"] == "large-v3-turbo"
    assert call["tolerance"] == 0.05 and call["chunk_seconds"] == 30.0
    assert call["workdir_existed"] is True and not call["workdir"].exists()
    assert call["options"] == {"device": "cpu", "compute_type": None, "language": "sv"}


@pytest.mark.parametrize("given", ["audio", "reference"])
def test_longform_needs_both_the_audio_and_its_reference(scoring, given):
    options = {"longform_audio": scoring.audio} if given == "audio" else {"longform_reference": scoring.reference}
    with pytest.raises(ValueError, match="both an audio file and a reference"):
        _run(scoring, **options)
    assert scoring.calls == []


def test_longform_bad_inputs_are_refused_before_any_model_is_scored(scoring, tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(scoring, longform_audio=tmp_path / "missing.wav", longform_reference=scoring.reference)

    empty = tmp_path / "empty.txt"
    empty.write_text(" ... ", encoding="utf-8")
    with pytest.raises(ValueError, match="no words"):
        _run(scoring, longform_audio=scoring.audio, longform_reference=empty)

    with pytest.raises(FileNotFoundError):
        _run(scoring, longform_audio=scoring.audio, longform_reference=tmp_path / "nope.txt")
    assert scoring.calls == [] and scoring.longform_calls == []


def test_longform_is_refused_with_the_hf_backend(scoring):
    with pytest.raises(ValueError, match="CT2"):
        _run(scoring, backend="hf", longform_audio=scoring.audio, longform_reference=scoring.reference)
    assert scoring.calls == []


# --- predictions and privacy --------------------------------------------------------------


def test_predictions_are_tagged_by_model_and_never_enter_the_gate(scoring):
    predictions = []

    result = _run(scoring, predictions=predictions, longform_audio=scoring.audio, longform_reference=scoring.reference)

    assert [(p["model"], p["reference"]) for p in predictions] == [
        ("candidate", SECRET_REFERENCE), ("baseline", SECRET_REFERENCE),
    ]
    written = (scoring.candidate_dir / "gate.json").read_text(encoding="utf-8")
    for text in (written, json.dumps(result, ensure_ascii=False)):
        assert "paimeneni" not in text and "paimenen" not in text and "spk" not in text


def test_no_predictions_are_collected_unless_asked_for(scoring):
    _run(scoring)
    assert [call["collecting"] for call in scoring.calls] == [False, False]

    _run(scoring, predictions=[])
    assert [call["collecting"] for call in scoring.calls[2:]] == [True, True]


# --- render_gate -----------------------------------------------------------------------


def test_render_gate_shows_metrics_and_verdicts_and_no_text(scoring):
    scoring.per_speaker = {"n_speakers": 5, "n_low_audio": 1, "unattributed_utterances": 0, "min_audio_seconds": 30.0,
                           "wer": _spread(0.05, 0.4), "wer_reliable": None}
    scoring.longform = _longform(regression=True, reasons=["coverage fell from 99.0% to 40.0%"])
    result = _run(scoring, longform_audio=scoring.audio, longform_reference=scoring.reference)

    text = "\n".join(gate.render_gate(result))

    assert "Dataset version v-1, split test: 40 utterances, 4.0 min of audio" in text
    assert "candidate" in text and "baseline" in text and "0.1000" in text and "0.2000" in text
    assert "+50.0%" in text
    assert "5 speakers, 1 with under 30 s" in text and "max 0.4000" in text and "enough audio: n/a" in text
    assert "Long-form (30.0 min, 4000 reference words)" in text
    assert "[PASS] beats_baseline_on_test" in text and "[FAIL] longform_no_regression" in text
    assert text.rstrip().endswith("Gate: FAILED")


def test_render_gate_marks_skipped_checks_and_copes_with_missing_numbers(scoring):
    scoring.wers["cand"] = None
    text = "\n".join(gate.render_gate(_run(scoring)))

    assert "n/a" in text and "[SKIP] longform_no_regression" in text and "[FAIL] beats_baseline_on_test" in text

    scoring.wers["cand"] = 0.1
    passed = "\n".join(gate.render_gate(_run(scoring)))
    assert passed.rstrip().endswith("Gate: PASSED") and "Candidate per-speaker" not in passed


# --- the CLI ---------------------------------------------------------------------------


def _eval_argv(scoring, *extra):
    return ["eval", "--model", str(scoring.candidate_dir), "--dataset", "dataset-dir", *extra]


def test_cli_exits_0_when_the_gate_passes_and_prints_the_verdict(scoring, capsys):
    assert cli.main(_eval_argv(scoring)) == 0

    out = capsys.readouterr().out
    assert "Gate: PASSED" in out and str(scoring.candidate_dir / "gate.json") in out
    assert json.loads((scoring.candidate_dir / "gate.json").read_text(encoding="utf-8"))["passed"] is True


def test_cli_exits_3_when_the_gate_fails(scoring, capsys):
    scoring.wers["cand"] = 0.4

    assert cli.main(_eval_argv(scoring)) == 3

    assert "Gate: FAILED" in capsys.readouterr().out
    assert json.loads((scoring.candidate_dir / "gate.json").read_text(encoding="utf-8"))["passed"] is False


def test_cli_require_longform_fails_a_gate_without_long_form_material(scoring):
    assert cli.main(_eval_argv(scoring)) == 0
    assert cli.main(_eval_argv(scoring, "--require-longform")) == 3


def test_cli_runs_the_longform_check_when_given_material(scoring, capsys):
    scoring.longform = _longform(regression=True, reasons=["coverage fell"])

    code = cli.main(_eval_argv(
        scoring, "--longform-audio", str(scoring.audio), "--longform-reference", str(scoring.reference),
        "--longform-tolerance", "0.05",
    ))

    assert code == 3
    assert scoring.longform_calls[0]["tolerance"] == 0.05
    assert "[FAIL] longform_no_regression" in capsys.readouterr().out


def test_cli_exits_1_on_bad_usage_without_writing_a_gate(scoring, capsys):
    assert cli.main(_eval_argv(scoring, "--longform-audio", str(scoring.audio))) == 1

    assert "error:" in capsys.readouterr().err
    assert not (scoring.candidate_dir / "gate.json").exists()
    assert scoring.calls == []


def test_cli_exits_1_when_a_model_cannot_be_loaded(scoring, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise ModelLoadError("Could not load model 'x' on any device")

    monkeypatch.setattr(gate, "evaluate_model", broken)

    assert cli.main(_eval_argv(scoring)) == 1

    assert "Could not load model" in capsys.readouterr().err
    assert not (scoring.candidate_dir / "gate.json").exists()


def test_cli_exits_1_when_the_candidate_is_not_a_directory_and_has_no_out(scoring, capsys):
    assert cli.main(["eval", "--model", "large-v3-turbo", "--dataset", "d"]) == 1
    assert "explicit output path" in capsys.readouterr().err


def test_cli_out_and_defaults(scoring, tmp_path, monkeypatch):
    seen = {}

    def recorder(candidate, dataset_dir, **options):
        seen.update(candidate=candidate, dataset_dir=dataset_dir, **options)
        return {"passed": True, "checks": [], "metrics": {"candidate": _scored(0.1), "baseline": _scored(0.2),
                                                          "relative_wer_improvement": 0.5},
                "dataset_version": "v", "split": "test", "longform": None}

    monkeypatch.setattr(gate, "evaluate_gate", recorder)

    assert cli.main(["eval", "--model", "m", "--dataset", "d", "--out", str(tmp_path / "x.json")]) == 0

    assert seen == {
        "candidate": "m", "dataset_dir": "d", "split": "test", "baseline": "large-v3-turbo", "backend": "ct2",
        "device": "auto", "compute_type": None, "language": "fi", "longform_audio": None, "longform_reference": None,
        "longform_tolerance": 0.02, "min_improvement": 0.0, "require_longform": False, "out": str(tmp_path / "x.json"),
        "predictions": None,
    }


def test_cli_parses_every_option(scoring, monkeypatch):
    seen = {}
    monkeypatch.setattr(gate, "evaluate_gate", lambda candidate, dataset_dir, **options: seen.update(options) or {
        "passed": True, "checks": [], "metrics": {"candidate": _scored(0.1), "baseline": _scored(0.2),
                                                  "relative_wer_improvement": 0.5},
        "dataset_version": None, "split": "dev", "longform": None})

    cli.main([
        "eval", "--model", "m", "--backend", "hf", "--dataset", "d", "--split", "dev", "--baseline", "models/base",
        "--language", "sv", "--device", "cpu", "--compute-type", "int8", "--longform-audio", "a.wav",
        "--longform-reference", "r.txt", "--longform-tolerance", "0.1", "--min-improvement", "0.05",
        "--require-longform", "--out", "g.json",
    ])

    assert seen["backend"] == "hf" and seen["split"] == "dev" and seen["baseline"] == "models/base"
    assert (seen["language"], seen["device"], seen["compute_type"]) == ("sv", "cpu", "int8")
    assert (seen["longform_audio"], seen["longform_reference"]) == ("a.wav", "r.txt")
    assert (seen["longform_tolerance"], seen["min_improvement"], seen["require_longform"]) == (0.1, 0.05, True)


@pytest.mark.parametrize("bad", [["--backend", "onnx"], ["--device", "tpu"], ["--min-improvement", "lots"]])
def test_cli_rejects_invalid_values(scoring, bad):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(_eval_argv(scoring, *bad))
    assert excinfo.value.code == 2


def test_cli_dumps_predictions_only_on_request_and_keeps_them_out_of_everything_else(scoring, tmp_path, capsys):
    cli.main(_eval_argv(scoring))
    assert not (tmp_path / "predictions.jsonl").exists()

    dump = tmp_path / "out" / "predictions.jsonl"
    assert cli.main(_eval_argv(scoring, "--dump-predictions", str(dump))) == 0

    rows = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    assert [r["model"] for r in rows] == ["candidate", "baseline"]
    assert rows[0]["reference"] == SECRET_REFERENCE and rows[0]["hypothesis"] == SECRET_HYPOTHESIS
    captured = capsys.readouterr()
    assert "paimeneni" not in captured.out + captured.err
    assert "paimeneni" not in (scoring.candidate_dir / "gate.json").read_text(encoding="utf-8")
    assert "including transcript text" in captured.out


def test_cli_checks_the_dump_path_before_evaluating(scoring, tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")

    assert cli.main(_eval_argv(scoring, "--dump-predictions", str(blocker / "sub" / "p.jsonl"))) == 1
    assert scoring.calls == []
    assert "error:" in capsys.readouterr().err


# --- end to end over a real dataset with stub backends -----------------------------------


def _write_dataset(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = [(1, "spk-a", "Hyvää päivää.", 1.0), (2, "spk-b", "Kiitos paljon.", 1.5), (3, "spk-b", "Herra on hyvä.", 2.0)]
    features = Features({"audio": Value("string"), "text": Value("string"), "duration": Value("float64"),
                         "speaker_id": Value("string"), "recording_id": Value("int64"), "quality_score": Value("float64")})
    items = []
    for recording_id, speaker, text, seconds in rows:
        path = audio_dir / f"{recording_id}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(np.zeros(int(seconds * RATE), dtype="<i2").tobytes())
        items.append({"audio": str(path), "text": text, "duration": seconds, "speaker_id": speaker,
                      "recording_id": recording_id, "quality_score": 1.0})
    split = Dataset.from_list(items, features=features).cast_column("audio", Audio(sampling_rate=RATE))
    DatasetDict({"train": split, "dev": split, "test": split}).save_to_disk(str(tmp_path / "dataset"))
    (tmp_path / "dataset" / "build_metadata.json").write_text(json.dumps({"dataset_version": "abc123"}))
    return tmp_path / "dataset"


def test_the_gate_end_to_end_with_stub_ct2_backends(tmp_path, monkeypatch, capsys):
    dataset = _write_dataset(tmp_path)
    candidate_dir = tmp_path / "ct2"
    candidate_dir.mkdir()
    answers = {"hyvää päivää": 10, "kiitos paljon": 15, "herra on hyvä": 20}
    by_duration = {tenths: text for text, tenths in answers.items()}
    events = []

    def fake_ct2(model, device, compute_type, language):
        events.append(f"open {Path(model).name}")
        if Path(model).name == "ct2":  # the candidate gets everything right
            transcribe = lambda samples: by_duration[round(len(samples) / RATE * 10)]  # noqa: E731
        else:  # the baseline garbles the last utterance
            transcribe = lambda samples: "väärin" if len(samples) > 1.9 * RATE else by_duration[round(len(samples) / RATE * 10)]  # noqa: E731
        return transcribe, lambda: events.append(f"close {Path(model).name}")

    monkeypatch.setattr(evaluate, "ct2_backend", fake_ct2)

    code = cli.main(["eval", "--model", str(candidate_dir), "--dataset", str(dataset), "--baseline", "zero-shot"])

    assert code == 0
    assert events == ["open ct2", "close ct2", "open zero-shot", "close zero-shot"]  # one model at a time
    written = json.loads((candidate_dir / "gate.json").read_text(encoding="utf-8"))
    assert written["passed"] is True and written["dataset_version"] == "abc123"
    assert written["metrics"]["candidate"]["wer_normalised"] == 0.0
    assert written["metrics"]["baseline"]["wer_normalised"] == pytest.approx(3 / 7, abs=1e-6)
    assert written["metrics"]["relative_wer_improvement"] == 1.0
    assert written["metrics"]["candidate"]["per_speaker"]["n_speakers"] == 2
    assert "Gate: PASSED" in capsys.readouterr().out
