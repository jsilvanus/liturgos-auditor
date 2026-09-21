"""The chunk core of batch transcription, shared by the job runner and by scripts.

`transcribe_chunk` turns one planned chunk into a result in ABSOLUTE media time.
The runner calls it once per chunk (through the batch queue, persisting each
result); `transcribe_file` is the same loop run synchronously start to finish
for callers that just want a transcript of a file, such as the evaluation
gate's long-form check.

Chunks are decoded independently (`condition_on_previous_text=False`): a
hallucination loop cannot outlive its chunk. Context instead crosses the
boundary explicitly, as the tail of the previous chunk's text in the prompt.
"""

from pathlib import Path

from ..audio import MediaDecodeError, normalize_to_wav
from .assemble import assemble
from .chunking import plan_chunks
from .wav import open_pcm16_mono

SNAP_WINDOW_SECONDS = 5.0


def plan_job_chunks(wav, chunk_seconds, find_gap=None):
    """plan_chunks with a snap window that also fits short chunks.

    The default +/- 5 s window needs chunk_seconds > 10; the API accepts down to
    5 s, so the window shrinks to a quarter of the chunk there.
    """
    window = min(SNAP_WINDOW_SECONDS, chunk_seconds / 4)
    return plan_chunks(wav, chunk_seconds, snap_window=window, find_gap=find_gap)


def carry_prompt(user_prompt, previous_text, max_chars):
    """The prompt for a chunk: the user's prompt plus the tail of the previous chunk's text.

    The tail is at most `max_chars` characters and starts on a word boundary
    (a cut word would only confuse the decoder). `max_chars` of 0 turns the
    carry-over off. Returns None when there is nothing to pass.
    """
    parts = [(user_prompt or "").strip()]
    text = (previous_text or "").strip()
    if max_chars > 0 and text:
        if len(text) > max_chars:
            tail = text[-max_chars:]
            if not (text[-max_chars - 1].isspace() or tail[0].isspace()):
                # The cut fell inside a word: drop the fragment (all of it if there is no boundary).
                pieces = tail.split(None, 1)
                tail = pieces[1] if len(pieces) == 2 else ""
            text = tail.strip()
        parts.append(text)
    return " ".join(part for part in parts if part) or None


def transcribe_chunk(host, wav, chunk, *, language, word_timestamps, prompt, vad, previous_text, carry_chars):
    """Transcribe one chunk of an open WAV; segment and word times are absolute (offset by chunk.start).

    Only this chunk's samples are ever read into memory. Blocking: the runner
    calls it in the inference queue's worker thread. The result names the model
    that produced it (`model`, when the host has a model_id), because a model
    switch between two chunks of one job is possible and must stay visible.
    """
    samples = wav.read(chunk.start_sample, chunk.end_sample)
    result = host.transcribe_array(
        samples,
        language,
        prompt=carry_prompt(prompt, previous_text, carry_chars),
        vad=vad,
        condition_on_previous_text=False,
        word_timestamps=word_timestamps,
        time_offset=chunk.start,
    )
    chunk_result = {"text": result["text"], "language": result.get("language"), "segments": result["segments"]}
    model_id = getattr(host, "model_id", None)
    if model_id:
        chunk_result["model"] = model_id
    return chunk_result


def transcribe_file(
    host,
    source_path,
    *,
    workdir,
    language="fi",
    chunk_seconds=60.0,
    word_timestamps=True,
    prompt=None,
    vad=True,
    carry_chars=200,
    normalize=normalize_to_wav,
    find_gap=None,
    on_chunk=None,
):
    """Transcribe a whole media file synchronously and return the assembled result.

    Same steps as a job (normalise, plan, transcribe each chunk in order,
    assemble) without persistence or a queue. `workdir` must exist; the
    normalised WAV lives there and is removed afterwards, the source is never
    touched. `on_chunk(index, total, result)` is called after every chunk.
    """
    wav_path = Path(workdir) / "audio.wav"
    try:
        normalize(source_path, wav_path)
        with open_pcm16_mono(wav_path) as wav:
            chunks = plan_job_chunks(wav, chunk_seconds, find_gap=find_gap)
            if not chunks:
                raise MediaDecodeError("The media contains no audio")
            manifest = {
                "params": {"language": language},
                "audio": {"duration_seconds": wav.duration, "sample_rate": wav.sample_rate},
                "chunks": [chunk.to_dict() for chunk in chunks],
            }

            results, previous_text = {}, None
            for chunk in chunks:
                result = transcribe_chunk(
                    host,
                    wav,
                    chunk,
                    language=language,
                    word_timestamps=word_timestamps,
                    prompt=prompt,
                    vad=vad,
                    previous_text=previous_text,
                    carry_chars=carry_chars,
                )
                results[chunk.index] = result
                previous_text = result["text"]
                if on_chunk is not None:
                    on_chunk(chunk.index, len(chunks), result)
        return assemble(manifest, results)
    finally:
        wav_path.unlink(missing_ok=True)
