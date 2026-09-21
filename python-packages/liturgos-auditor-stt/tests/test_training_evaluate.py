import json
import sys
import types
import wave

import numpy as np
import pytest
from datasets import Audio, Dataset, DatasetDict, Features, Value

from auditor_stt.serve.model import ModelLoadError
from auditor_stt.training import evaluate

RATE = 16000

# Text and speakers of the synthetic test split; each utterance has its own duration so a stub
# "model" can tell utterances apart by the length of the samples it is handed.
TEST_ROWS = [
    # (recording_id, speaker, text, seconds)
    (1, "spk-a", "Hyvää päivää, Ärjä.", 1.0),
    (2, "spk-a", "Kiitos paljon.", 1.5),
    (3, "spk-b", "Herra on minun paimeneni.", 2.0),
    (4, "", "Tämä lause on ilman puhujaa.", 2.5),
]
ANSWERS = {  # what a perfect model prints, keyed by duration in tenths of a second
    10: "hyvää päivää ärjä",  # differs from the reference only in case and punctuation
    15: "kiitos paljon",
    20: "Herra on minun paimeneni.",
    25: "Tämä lause on ilman puhujaa.",
}


def _write_wav(path, seconds):
    t = np.arange(int(seconds * RATE)) / RATE
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes())


def _make_dataset(tmp_path, rows=TEST_ROWS, version="v-test", speakers=True):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    columns = {
        "audio": Value("string"),
        "text": Value("string"),
        "duration": Value("float64"),
        "recording_id": Value("int64"),
        "quality_score": Value("float64"),
    }
    if speakers:
        columns["speaker_id"] = Value("string")
    features = Features(columns)

    def split(split_rows):
        items = []
        for recording_id, speaker, text, seconds in split_rows:
            path = audio_dir / f"{recording_id}.wav"
            _write_wav(path, seconds)
            item = {"audio": str(path), "text": text, "duration": seconds, "recording_id": recording_id, "quality_score": 1.0}
            if speakers:
                item["speaker_id"] = speaker
            items.append(item)
        return Dataset.from_list(items, features=features).cast_column("audio", Audio(sampling_rate=RATE))

    train = split([(100, "spk-t", "Koulutuslause.", 1.0)])
    DatasetDict({"train": train, "dev": train, "test": split(rows)}).save_to_disk(str(tmp_path / "dataset"))
    if version:
        (tmp_path / "dataset" / "build_metadata.json").write_text(json.dumps({"dataset_version": version}))
    return tmp_path / "dataset"


def _perfect(samples):
    return ANSWERS[round(len(samples) / RATE * 10)]


# --- evaluate_dataset ----------------------------------------------------------------


def test_evaluate_dataset_scores_the_split_and_records_the_dataset_version(tmp_path):
    result = evaluate.evaluate_dataset(_perfect, _make_dataset(tmp_path))

    assert result["split"] == "test"
    assert result["dataset_version"] == "v-test"
    assert result["n"] == 4
    assert result["audio_seconds"] == pytest.approx(7.0, abs=0.05)
    assert result["wer_normalised"] == 0.0 and result["cer_normalised"] == 0.0
    assert result["wer_raw"] > 0  # the first answer lacks the capitals and punctuation of its label
    assert result["reference_words"] == 3 + 2 + 4 + 5


def test_evaluate_dataset_reports_the_per_speaker_spread_without_speaker_ids_or_text(tmp_path):
    def wrong_for_b(samples):
        return "väärin" if round(len(samples) / RATE * 10) == 20 else _perfect(samples)

    result = evaluate.evaluate_dataset(wrong_for_b, _make_dataset(tmp_path))

    speakers = result["per_speaker"]
    assert speakers["n_speakers"] == 2  # the row with an empty speaker id is not a speaker
    assert speakers["unattributed_utterances"] == 1
    assert speakers["wer"]["min"] == 0.0 and speakers["wer"]["max"] == 1.0
    assert "speakers" not in speakers
    stored = json.dumps(result, ensure_ascii=False)
    assert "spk-a" not in stored and "paimeneni" not in stored and "väärin" not in stored


def test_evaluate_dataset_hands_predictions_only_to_a_list_the_caller_supplies(tmp_path):
    dataset = _make_dataset(tmp_path)
    predictions = []

    result = evaluate.evaluate_dataset(_perfect, dataset, predictions=predictions)

    assert [p["recording_id"] for p in predictions] == [1, 2, 3, 4]
    assert predictions[0] == {
        "recording_id": 1, "speaker_id": "spk-a", "reference": "Hyvää päivää, Ärjä.", "hypothesis": "hyvää päivää ärjä",
    }
    assert "Ärjä" not in json.dumps(result, ensure_ascii=False)  # the result stays text-free


def test_evaluate_dataset_can_skip_the_speaker_breakdown(tmp_path):
    result = evaluate.evaluate_dataset(_perfect, _make_dataset(tmp_path), speaker_ids=False)
    assert result["per_speaker"] is None


def test_evaluate_dataset_copes_with_a_dataset_without_speakers_or_build_metadata(tmp_path):
    dataset = _make_dataset(tmp_path, speakers=False, version=None)
    predictions = []

    result = evaluate.evaluate_dataset(_perfect, dataset, predictions=predictions)

    assert result["per_speaker"] is None
    assert result["dataset_version"] is None
    assert predictions[0]["speaker_id"] is None


def test_evaluate_dataset_treats_a_missing_transcript_as_empty(tmp_path):
    result = evaluate.evaluate_dataset(lambda samples: None, _make_dataset(tmp_path))
    assert result["wer_normalised"] == 1.0


def test_evaluate_dataset_can_score_another_split(tmp_path):
    dataset = _make_dataset(tmp_path)
    result = evaluate.evaluate_dataset(lambda samples: "koulutuslause", dataset, split="dev")
    assert result["split"] == "dev" and result["n"] == 1 and result["wer_normalised"] == 0.0


def test_evaluate_dataset_rejects_a_missing_split_or_a_dataset_without_splits(tmp_path):
    dataset = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match="'holdout' split"):
        evaluate.evaluate_dataset(_perfect, dataset, split="holdout")

    Dataset.from_dict({"text": ["x"]}).save_to_disk(str(tmp_path / "single"))
    with pytest.raises(ValueError, match="'test' split"):
        evaluate.evaluate_dataset(_perfect, tmp_path / "single")


def test_audio_array_decodes_embedded_bytes_and_paths(tmp_path):
    path = tmp_path / "tone.wav"
    _write_wav(path, 0.5)

    from_path = evaluate.audio_array({"bytes": None, "path": str(path)})
    from_bytes = evaluate.audio_array({"bytes": path.read_bytes(), "path": None})

    assert from_path.dtype == np.float32 and abs(len(from_path) - RATE // 2) <= 1
    np.testing.assert_allclose(from_path, from_bytes, atol=1e-6)


# --- evaluate_model ------------------------------------------------------------------


def test_evaluate_model_frees_the_model_even_when_scoring_fails(tmp_path, monkeypatch):
    events = []

    def failing(samples):
        raise RuntimeError("decoder crashed")

    monkeypatch.setattr(evaluate, "ct2_backend", lambda *a, **k: (failing, lambda: events.append("closed")))

    with pytest.raises(RuntimeError, match="decoder crashed"):
        evaluate.evaluate_model("some-model", _make_dataset(tmp_path))
    assert events == ["closed"]


def test_evaluate_model_checks_the_dataset_before_loading_a_model(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(evaluate, "ct2_backend", lambda *a, **k: opened.append(a) or (_perfect, lambda: None))

    with pytest.raises(ValueError, match="'nothing' split"):
        evaluate.evaluate_model("some-model", _make_dataset(tmp_path), "nothing")
    assert opened == []


def test_evaluate_model_forwards_backend_options(tmp_path, monkeypatch):
    seen = {}

    def fake_ct2(model, device, compute_type, language):
        seen.update(model=model, device=device, compute_type=compute_type, language=language)
        return _perfect, lambda: None

    monkeypatch.setattr(evaluate, "ct2_backend", fake_ct2)
    evaluate.evaluate_model("m", _make_dataset(tmp_path), device="cpu", compute_type="int8", language="sv")
    assert seen == {"model": "m", "device": "cpu", "compute_type": "int8", "language": "sv"}


def test_open_backend_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        evaluate.open_backend("onnx", "m")


# --- the ct2 backend (through ModelHost) -----------------------------------------------


class _StubHost:
    instances = []

    def __init__(self, model_id, model_dir=None, device="auto", compute_type=None):
        self.model_id, self.model_dir = model_id, model_dir
        self.requested = (device, compute_type)
        self.device, self.compute_type = "cpu", "int8"
        self.model, self.loaded = None, False
        self.calls = []
        _StubHost.instances.append(self)

    def load(self):
        self.model, self.loaded = object(), True

    def transcribe_array(self, samples, language=None, **options):
        self.calls.append((len(samples), language, options))
        return {"text": "  hei maailma ", "language": language, "segments": []}


@pytest.fixture
def stub_host(monkeypatch):
    _StubHost.instances = []
    monkeypatch.setattr(evaluate, "ModelHost", _StubHost)
    monkeypatch.delenv("AUDITOR_STT_MODEL_DIR", raising=False)
    return _StubHost


def test_ct2_backend_transcribes_through_model_host_transcribe_array(stub_host):
    transcribe, close = evaluate.ct2_backend("large-v3-turbo", device="cuda", compute_type="float16", language="fi")

    (host,) = stub_host.instances
    assert host.loaded and host.model_id == "large-v3-turbo" and host.requested == ("cuda", "float16")
    assert transcribe(np.zeros(RATE, dtype=np.float32)) == "  hei maailma "
    assert host.calls == [(RATE, "fi", {"word_timestamps": False})]  # no timestamps needed to score text
    assert (transcribe.device, transcribe.compute_type) == ("cpu", "int8")
    close()


def test_ct2_backend_close_frees_the_model(stub_host):
    transcribe, close = evaluate.ct2_backend("large-v3-turbo")
    (host,) = stub_host.instances

    close()

    assert host.model is None and host.loaded is False


def test_ct2_models_can_be_loaded_one_after_the_other(stub_host):
    first, close_first = evaluate.ct2_backend("first")
    close_first()
    second, close_second = evaluate.ct2_backend("second")

    first_host, second_host = stub_host.instances
    assert first_host.loaded is False and second_host.loaded is True
    close_second()


def test_ct2_backend_shares_the_services_download_cache(stub_host, monkeypatch):
    monkeypatch.setenv("AUDITOR_STT_MODEL_DIR", "/models-cache")
    evaluate.ct2_backend("large-v3-turbo")
    assert stub_host.instances[0].model_dir == "/models-cache"


def test_ct2_backend_accepts_an_exported_model_directory(stub_host, tmp_path):
    (tmp_path / "model.bin").write_bytes(b"x")
    evaluate.ct2_backend(tmp_path)
    assert stub_host.instances[0].model_id == str(tmp_path)


def test_ct2_backend_points_a_hugging_face_directory_at_the_right_backend(stub_host, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="not a CTranslate2 model directory.*--backend hf"):
        evaluate.ct2_backend(tmp_path)
    assert stub_host.instances == []


def test_ct2_backend_surfaces_a_model_that_will_not_load(monkeypatch):
    class _Failing(_StubHost):
        def load(self):
            raise ModelLoadError("Could not load model 'x' on any device")

    monkeypatch.setattr(evaluate, "ModelHost", _Failing)
    with pytest.raises(ModelLoadError):
        evaluate.ct2_backend("x")


# --- the hf backend ------------------------------------------------------------------


def test_hf_backend_refuses_a_lora_adapter_with_a_pointer_to_the_merged_model(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")

    with pytest.raises(ValueError, match="LoRA adapter.*_merged_hf"):
        evaluate.hf_backend(tmp_path)


# --- timestamp sanity and the long-form comparison -----------------------------------


def _segments(*spans):
    return [{"start": start, "end": end} for start, end in spans]


def test_timestamp_sanity_of_well_formed_segments():
    sanity = evaluate.timestamp_sanity(_segments((0, 4), (4, 9), (10, 16), (17, 20)), 20.0)

    assert sanity == {
        "n_segments": 4,
        "order_violations": 0,
        "long_segment_fraction": 0.0,
        "long_segment_seconds": evaluate.LONG_SEGMENT_SECONDS,
        "coverage": 1.0,
        "segments_per_minute": 12.0,
    }


def test_timestamp_sanity_counts_overlaps_and_backwards_segments():
    # the second overlaps the first by 1 s, the third ends before it starts; touching (4.0 -> 4.03) is fine
    sanity = evaluate.timestamp_sanity(_segments((0, 5), (4, 8), (10, 9), (9.0, 12), (12.03, 14)), 14.0)
    assert sanity["order_violations"] == 2


def test_timestamp_sanity_flags_window_long_segments_and_missing_coverage():
    sanity = evaluate.timestamp_sanity(_segments((0, 30), (30, 60), (60, 63), (63, 66)), 120.0)

    assert sanity["long_segment_fraction"] == 0.5
    assert sanity["coverage"] == 0.55
    assert sanity["segments_per_minute"] == 2.0


def test_timestamp_sanity_without_segments_or_duration():
    empty = evaluate.timestamp_sanity([], 60.0)
    assert (empty["n_segments"], empty["coverage"], empty["long_segment_fraction"], empty["segments_per_minute"]) == (0, 0.0, 0.0, 0.0)
    assert evaluate.timestamp_sanity(_segments((0, 1)), 0.0)["coverage"] == 0.0


def _run(wer, *, violations=0, long=0.0, coverage=0.98):
    return {"wer": wer, "words": 1000, "sanity": {"order_violations": violations, "long_segment_fraction": long, "coverage": coverage}}


def test_compare_longform_flags_a_wer_regression_beyond_the_tolerance():
    regression, reasons = evaluate.compare_longform(_run(0.23), _run(0.20), 0.02)
    assert regression and len(reasons) == 1 and "0.0300 above" in reasons[0]


def test_compare_longform_tolerates_a_wer_increase_up_to_the_tolerance():
    assert evaluate.compare_longform(_run(0.22), _run(0.20), 0.02) == (False, [])
    assert evaluate.compare_longform(_run(0.50), _run(0.48), 0.02) == (False, [])  # float noise must not tip it over
    assert evaluate.compare_longform(_run(0.10), _run(0.20), 0.02) == (False, [])  # better is fine


def test_compare_longform_flags_more_order_violations_than_the_baseline():
    assert evaluate.compare_longform(_run(0.2, violations=3), _run(0.2, violations=1), 0.02)[0] is True
    assert evaluate.compare_longform(_run(0.2, violations=1), _run(0.2, violations=1), 0.02)[0] is False
    assert evaluate.compare_longform(_run(0.2, violations=0), _run(0.2, violations=2), 0.02)[0] is False


def test_compare_longform_flags_long_segments_that_more_than_double():
    regression, reasons = evaluate.compare_longform(_run(0.2, long=0.30), _run(0.2, long=0.10), 0.02)
    assert regression and "window-long" in reasons[0]
    assert evaluate.compare_longform(_run(0.2, long=0.15), _run(0.2, long=0.10), 0.02)[0] is False  # not doubled
    assert evaluate.compare_longform(_run(0.2, long=0.02), _run(0.2, long=0.0), 0.02)[0] is False  # too few to matter
    assert evaluate.compare_longform(_run(0.2, long=0.10), _run(0.2, long=0.0), 0.02)[0] is True


def test_compare_longform_flags_a_coverage_drop_of_more_than_ten_points():
    regression, reasons = evaluate.compare_longform(_run(0.2, coverage=0.80), _run(0.2, coverage=0.95), 0.02)
    assert regression and "coverage" in reasons[0]
    assert evaluate.compare_longform(_run(0.2, coverage=0.90), _run(0.2, coverage=0.95), 0.02)[0] is False


def test_compare_longform_lists_every_reason():
    regression, reasons = evaluate.compare_longform(
        _run(0.30, violations=2, long=0.5, coverage=0.5), _run(0.20), 0.02
    )
    assert regression and len(reasons) == 4


# --- longform_check (stub hosts, stubbed transcribe_file) --------------------------------

REFERENCE = "Herra on minun paimeneni, ei minulta mitään puutu. Hän kaitsee minua vihreillä niityillä."
SECRET_HYPOTHESIS = "herra on minun paimeneni ei minulta mitään puutu hän kaitsee"


@pytest.fixture
def longform(monkeypatch, tmp_path):
    """Stub hosts and a stub transcribe_file whose result depends on which model runs."""
    events, results = [], {}

    def load(model, device="auto", compute_type=None):
        events.append(f"load {model}")
        return types.SimpleNamespace(name=model, device=device, compute_type=compute_type)

    def free(host):
        events.append(f"free {host.name}")

    def fake_transcribe_file(host, source_path, *, workdir, language="fi", chunk_seconds=60.0, **rest):
        events.append(f"transcribe {host.name}")
        events.append(("args", str(source_path), language, chunk_seconds, rest))
        return results[host.name]

    monkeypatch.setattr(evaluate, "load_ct2_host", load)
    monkeypatch.setattr(evaluate, "free_host", free)
    monkeypatch.setattr(evaluate, "transcribe_file", fake_transcribe_file)

    reference = tmp_path / "reference.txt"
    reference.write_text(REFERENCE, encoding="utf-8")
    workdir = tmp_path / "work"
    workdir.mkdir()
    return types.SimpleNamespace(events=events, results=results, reference=reference, workdir=workdir, audio=tmp_path / "sermon.wav")


def _result(text, segments, duration=180.0):
    return {"text": text, "segments": [{"start": s, "end": e, "text": "x"} for s, e in segments], "duration_seconds": duration}


GOOD_SEGMENTS = [(0, 8), (8, 20), (20, 33), (33, 60), (60, 75), (75, 100), (100, 130), (130, 160), (160, 178)]


def test_longform_check_runs_the_models_one_after_the_other(longform):
    longform.results["candidate"] = _result(SECRET_HYPOTHESIS, GOOD_SEGMENTS)
    longform.results["baseline"] = _result(SECRET_HYPOTHESIS, GOOD_SEGMENTS)

    evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)

    order = [e for e in longform.events if isinstance(e, str)]
    assert order == [
        "load candidate", "transcribe candidate", "free candidate",
        "load baseline", "transcribe baseline", "free baseline",
    ]


def test_longform_check_frees_the_model_when_transcription_fails(longform, monkeypatch):
    def broken(host, *args, **kwargs):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(evaluate, "transcribe_file", broken)

    with pytest.raises(RuntimeError, match="decode failed"):
        evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)
    assert longform.events == ["load candidate", "free candidate"]  # the baseline never loads


def test_longform_check_passes_the_pipeline_settings_through(longform):
    longform.results["candidate"] = longform.results["baseline"] = _result(SECRET_HYPOTHESIS, GOOD_SEGMENTS)

    evaluate.longform_check(
        "candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir,
        chunk_seconds=30.0, language="sv", device="cpu", compute_type="int8",
    )

    args = [e for e in longform.events if isinstance(e, tuple)]
    assert args[0] == ("args", str(longform.audio), "sv", 30.0, {})
    assert len(args) == 2


def test_longform_check_detects_a_regression_and_reports_metrics_only(longform):
    longform.results["baseline"] = _result(SECRET_HYPOTHESIS, GOOD_SEGMENTS)
    # the candidate loses most of the words and stops emitting timestamps: window-long segments, early end
    longform.results["candidate"] = _result("herra on", [(0, 30), (30, 60), (60, 90)], duration=180.0)

    result = evaluate.longform_check(
        "candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir, tolerance=0.02
    )

    assert result["regression"] is True
    assert result["candidate"]["wer"] > result["baseline"]["wer"] + 0.02
    assert result["wer_delta"] == pytest.approx(result["candidate"]["wer"] - result["baseline"]["wer"], abs=1e-6)
    assert result["candidate"]["sanity"]["long_segment_fraction"] == 1.0
    assert result["candidate"]["sanity"]["coverage"] == 0.5
    assert len(result["reasons"]) >= 3
    assert result["reference_words"] == 13 and result["audio_seconds"] == 180.0
    assert result["tolerance"] == 0.02 and result["chunk_seconds"] == 60.0
    stored = json.dumps(result, ensure_ascii=False)
    assert "paimeneni" not in stored and "herra" not in stored.lower()


def test_longform_check_reports_no_regression_for_an_equal_or_better_candidate(longform):
    longform.results["baseline"] = _result("herra on minun paimeneni", GOOD_SEGMENTS)
    longform.results["candidate"] = _result(SECRET_HYPOTHESIS, GOOD_SEGMENTS)

    result = evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)

    assert result["regression"] is False and result["reasons"] == []
    assert result["wer_delta"] < 0


def test_longform_check_tolerates_a_small_wer_increase(longform):
    tail = " minua vihreillä niityillä"
    longform.results["baseline"] = _result(SECRET_HYPOTHESIS + tail, GOOD_SEGMENTS)  # every word right
    longform.results["candidate"] = _result(SECRET_HYPOTHESIS + tail.replace("vihreillä", "vihreällä"), GOOD_SEGMENTS)  # 1 of 13 wrong

    tolerant = evaluate.longform_check(
        "candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir, tolerance=0.2
    )
    strict = evaluate.longform_check(
        "candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir, tolerance=0.0
    )

    assert tolerant["regression"] is False
    assert strict["regression"] is True


def test_longform_check_compares_normalised_text(longform):
    # capitals, commas and full stops must not count against either model
    longform.results["candidate"] = longform.results["baseline"] = _result(REFERENCE.upper(), GOOD_SEGMENTS)

    result = evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)

    assert result["candidate"]["wer"] == 0.0


def test_longform_check_reads_a_reference_with_a_byte_order_mark(longform):
    longform.reference.write_bytes(b"\xef\xbb\xbf" + REFERENCE.encode("utf-8"))
    longform.results["candidate"] = longform.results["baseline"] = _result(REFERENCE, GOOD_SEGMENTS)

    result = evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)

    assert result["candidate"]["wer"] == 0.0


def test_longform_check_refuses_audio_in_which_the_baseline_finds_no_speech(longform):
    longform.results["candidate"] = _result("", [])
    longform.results["baseline"] = _result("", [])

    with pytest.raises(ValueError, match="found no speech"):
        evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)
    assert longform.events[-1] == "free baseline"


def test_longform_check_counts_a_silent_candidate_as_a_regression(longform):
    longform.results["candidate"] = _result("", [])
    longform.results["baseline"] = _result(SECRET_HYPOTHESIS, GOOD_SEGMENTS)

    result = evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)

    assert result["regression"] is True
    assert result["candidate"]["sanity"]["coverage"] == 0.0 and result["candidate"]["wer"] == 1.0


def test_longform_check_rejects_an_empty_reference_before_loading_anything(longform):
    longform.reference.write_text(" \n ... \n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no words"):
        evaluate.longform_check("candidate", "baseline", longform.audio, longform.reference, workdir=longform.workdir)
    assert longform.events == []


def test_load_ct2_host_is_the_function_the_long_form_check_uses(stub_host):
    host = evaluate.load_ct2_host("large-v3-turbo", "cpu", "int8")
    assert host.loaded and host.requested == ("cpu", "int8")
    evaluate.free_host(host)
    assert host.model is None and host.loaded is False


def test_load_ct2_host_seeds_the_random_generator_before_every_model_loads(stub_host, monkeypatch):
    # Unseeded, faster-whisper's temperature fallback made the same model score differently on the
    # same split in different processes, so a gate could pass on one run and fail on the next.
    import ctranslate2

    events = []
    monkeypatch.setattr(ctranslate2, "set_random_seed", lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(stub_host, "load", lambda self: events.append(("load", self.model_id)))

    evaluate.load_ct2_host("candidate", "cpu", "int8")
    evaluate.load_ct2_host("baseline", "cpu", "int8")

    assert events == [
        ("seed", evaluate.EVAL_RANDOM_SEED), ("load", "candidate"),
        ("seed", evaluate.EVAL_RANDOM_SEED), ("load", "baseline"),
    ]


# --- the legacy entry point ------------------------------------------------------------


def test_legacy_main_scores_a_checkpoint_and_still_writes_test_metrics_json(tmp_path, monkeypatch, capsys):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    dataset = _make_dataset(tmp_path)
    seen, closed = {}, []

    def transcribe(samples):
        return _perfect(samples)

    transcribe.device = "cpu"

    def fake_hf(model, language="fi", device="auto"):
        seen.update(model=str(model), language=language)
        return transcribe, lambda: closed.append(True)

    monkeypatch.setattr(evaluate, "hf_backend", fake_hf)

    evaluate.main(["--model", str(model_dir), "--dataset", str(dataset), "--batch-size", "2", "--language", "fi"])

    assert seen == {"model": str(model_dir), "language": "fi"} and closed == [True]
    written = json.loads((model_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert written["model"] == str(model_dir.resolve())
    assert written["dataset"] == str(dataset.resolve())
    assert written["test_examples"] == 4 and written["device"] == "cpu"
    assert written["wer"] == written["wer_raw"] > 0  # `wer` keeps its old raw meaning
    assert written["wer_normalised"] == 0.0 and written["cer_normalised"] == 0.0
    assert written["dataset_version"] == "v-test"
    assert json.loads(capsys.readouterr().out) == written
    assert "paimeneni" not in json.dumps(written)


def test_legacy_main_refuses_a_lora_adapter(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="_merged_hf"):
        evaluate.main(["--model", str(tmp_path), "--dataset", str(tmp_path)])


# --- the real hf backend on a tiny random Whisper (needs torch; skipped where it cannot load) ---


def _torch_or_skip():
    try:
        import torch
        from transformers import WhisperConfig, WhisperForConditionalGeneration
    except Exception as exc:  # ImportError, or OSError when a host policy blocks the torch DLLs
        pytest.skip(f"torch and transformers unavailable: {exc}")
    # Importing a model class makes transformers replace its sys.modules entry, so a module
    # object bound before that is stale: patch the live one, which is what `from transformers import` reads.
    return torch, sys.modules["transformers"], WhisperConfig, WhisperForConditionalGeneration


def _tiny_whisper(tmp_path):
    torch, transformers, WhisperConfig, WhisperForConditionalGeneration = _torch_or_skip()
    config = WhisperConfig(
        vocab_size=100, num_mel_bins=8, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_source_positions=30, max_target_positions=20, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, decoder_start_token_id=3,
    )
    torch.manual_seed(0)
    model_dir = tmp_path / "tiny"
    WhisperForConditionalGeneration(config).save_pretrained(model_dir)
    return torch, transformers, model_dir


def test_hf_backend_generates_with_a_real_model_and_frees_it(tmp_path, monkeypatch):
    torch, transformers, model_dir = _tiny_whisper(tmp_path)
    fed = []

    class _Processor:
        # No tokenizer without a download: features are fixed-size zeros, and decoding names the token count.
        @staticmethod
        def from_pretrained(source, **kwargs):
            return _Processor()

        def feature_extractor(self, samples, sampling_rate, return_tensors):
            fed.append((len(samples), sampling_rate, return_tensors))
            return types.SimpleNamespace(input_features=torch.zeros(1, 8, 60))

        def batch_decode(self, ids, skip_special_tokens):
            assert skip_special_tokens is True and ids.dim() == 2 and ids.shape[0] == 1
            return [f" {ids.shape[1]} tokens "]

    monkeypatch.setattr(transformers, "WhisperProcessor", _Processor)

    transcribe, close = evaluate.hf_backend(model_dir, device="cpu", max_new_tokens=4)
    text = transcribe(np.zeros(RATE, dtype=np.float32))
    close()

    assert fed == [(RATE, RATE, "pt")]
    assert text.endswith("tokens") and not text.startswith(" ")  # stripped
    assert transcribe.device == "cpu"


def test_evaluate_dataset_runs_on_the_real_hf_backend(tmp_path, monkeypatch):
    torch, transformers, model_dir = _tiny_whisper(tmp_path)

    class _Processor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            return _Processor()

        def feature_extractor(self, samples, sampling_rate, return_tensors):
            return types.SimpleNamespace(input_features=torch.zeros(1, 8, 60))

        def batch_decode(self, ids, skip_special_tokens):
            return ["sana"]

    monkeypatch.setattr(transformers, "WhisperProcessor", _Processor)
    dataset = _make_dataset(tmp_path)

    transcribe, close = evaluate.hf_backend(model_dir, device="cpu", max_new_tokens=3)
    try:
        result = evaluate.evaluate_dataset(transcribe, dataset)
    finally:
        close()

    assert result["n"] == 4 and result["wer_normalised"] is not None


def test_audio_array_agrees_with_the_training_helper(tmp_path):
    _torch_or_skip()
    from auditor_stt.training.train import audio_array as train_audio_array

    path = tmp_path / "tone.wav"
    _write_wav(path, 0.5)
    audio = {"bytes": None, "path": str(path)}
    np.testing.assert_allclose(evaluate.audio_array(audio), train_audio_array(audio, 16000))
