"""faster-whisper model loading with GPU auto-detect and CPU int8 fallback."""

import logging

# PyAV is faster-whisper's own decoder dependency; its errors are how an
# undecodable upload surfaces (av.error.InvalidDataError is not an OSError).
from av.error import FFmpegError

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    pass


class AudioDecodeError(Exception):
    """The uploaded bytes could not be decoded as audio (client error, not a server fault)."""


class ModelHost:
    """Lazily loads and holds a single faster-whisper WhisperModel instance.

    Device selection tries CUDA/float16 first (unless a specific device was
    requested), falling back to CPU/int8 — both are inside the STT latency
    budget (see docs/plans/plan_local_stt.md in live-captions-yt), so silently degrading to CPU
    rather than failing startup is the right default for `device="auto"`.
    """

    def __init__(self, model_id, model_dir=None, device="auto", compute_type=None):
        self.model_id = model_id
        self.model_dir = model_dir
        self._requested_device = device
        self._requested_compute_type = compute_type
        self.device = None
        self.compute_type = None
        self.model = None
        self.loaded = False

    def load(self):
        from faster_whisper import WhisperModel

        last_error = None
        for device, compute_type in self._device_attempts():
            try:
                logger.info("Loading faster-whisper model %s on %s/%s", self.model_id, device, compute_type)
                self.model = WhisperModel(
                    self.model_id,
                    device=device,
                    compute_type=compute_type,
                    download_root=self.model_dir,
                )
                self.device = device
                self.compute_type = compute_type
                self.loaded = True
                return
            except Exception as exc:  # noqa: BLE001 - any backend init failure should fall through to the next device
                last_error = exc
                logger.warning("Failed to load model on %s/%s: %s", device, compute_type, exc)
        raise ModelLoadError(f"Could not load model '{self.model_id}' on any device") from last_error

    def _device_attempts(self):
        if self._requested_device == "cuda":
            return [("cuda", self._requested_compute_type or "float16")]
        if self._requested_device == "cpu":
            return [("cpu", self._requested_compute_type or "int8")]
        return [
            ("cuda", self._requested_compute_type or "float16"),
            ("cpu", self._requested_compute_type or "int8"),
        ]

    def transcribe(
        self,
        audio_path,
        language=None,
        *,
        prompt=None,
        vad=False,
        temperature=None,
        condition_on_previous_text=None,
        word_timestamps=True,
    ):
        """Transcribe an audio file (any container PyAV can decode)."""
        options = _decode_options(prompt, vad, temperature, condition_on_previous_text, word_timestamps)
        return self._transcribe(audio_path, language, options, time_offset=0.0)

    def transcribe_array(
        self,
        samples,
        language=None,
        *,
        prompt=None,
        vad=False,
        temperature=None,
        condition_on_previous_text=None,
        word_timestamps=True,
        time_offset=0.0,
    ):
        """Transcribe float32 mono 16 kHz samples, skipping the temp file and decode.

        `time_offset` (seconds) is added to every segment and word timestamp, so a
        chunk sliced out of a longer recording reports times on the recording's clock.
        """
        options = _decode_options(prompt, vad, temperature, condition_on_previous_text, word_timestamps)
        return self._transcribe(samples, language, options, time_offset=time_offset)

    def _transcribe(self, audio, language, options, time_offset):
        if not self.loaded:
            raise RuntimeError("Model is not loaded yet")
        try:
            segments, info = self.model.transcribe(audio, language=language, **options)
            segments = list(segments)
        except FFmpegError as exc:
            # Fixed message on purpose: the raw error embeds the server's temp file path.
            raise AudioDecodeError("Could not decode audio") from exc
        text = "".join(segment.text for segment in segments).strip()
        return {
            "text": text,
            "language": (info.language if info else None) or language,
            "segments": [_shape_segment(s, time_offset) for s in segments],
        }


def _decode_options(prompt, vad, temperature, condition_on_previous_text, word_timestamps):
    # Only override faster-whisper's own defaults when asked to, so an option left
    # unset keeps today's decoding behaviour exactly.
    options = {"word_timestamps": word_timestamps}
    if prompt:
        options["initial_prompt"] = prompt
    if vad:
        options["vad_filter"] = True
    if temperature is not None:
        options["temperature"] = temperature
    if condition_on_previous_text is not None:
        options["condition_on_previous_text"] = condition_on_previous_text
    return options


def _shape_segment(segment, time_offset):
    return {
        "start": segment.start + time_offset,
        "end": segment.end + time_offset,
        "text": segment.text.strip(),
        # Additive: consumers (saarnavideo) derive a per-segment confidence from these.
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
        "words": [
            {
                "start": word.start + time_offset,
                "end": word.end + time_offset,
                "text": word.word,
                "probability": word.probability,
            }
            for word in (segment.words or [])
        ],
    }
