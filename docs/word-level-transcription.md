# Word-level transcription

The live route `/inference`, the OpenAI-compatible route `/v1/audio/transcriptions` (with `response_format=verbose_json`) and the batch jobs API return segment timestamps plus word timestamps. Live requests always ask faster-whisper for word timestamps; there is no option to switch them off on the live routes. Batch jobs can (`word_timestamps=false`).

Which route returns what:

| Route | `response_format` | Body |
|-------|-------------------|------|
| `/inference` | `json` (default) or `verbose_json` | Full result: `text`, `language`, `segments[]` with words |
| `/inference` | `text` | Plain text of the transcript |
| `/v1/audio/transcriptions` | `json` (default) | Only `{"text": "..."}` |
| `/v1/audio/transcriptions` | `verbose_json` | The same full result as `/inference` |
| `/v1/audio/transcriptions` | `text` | Plain text of the transcript |
| both | anything else (including `vtt`, `srt`) | `400` |
| `/v1/jobs/{id}/result` | `format=json` | Full result plus job fields (see [Batch jobs](#batch-jobs)) |

Caption formats (VTT, SRT, YouTube Live Captions, plain text) are produced only by the jobs result route.

## Response format

Each segment has:

- `start`, `end`: segment timing in seconds
- `text`: segment text
- `avg_logprob`: faster-whisper's average log-probability of the segment's tokens
- `no_speech_prob`: faster-whisper's no-speech probability for the segment
- `words[]`: word timing records (empty when word timestamps are off)

`avg_logprob` and `no_speech_prob` are passed through as faster-whisper reports them and may be `null` if the model host does not provide them.

Each word record has:

- `start`, `end`: word timing in seconds
- `text`: the word as faster-whisper returns it. The service does not trim it, and it normally starts with a space
- `probability`: faster-whisper's probability for the word (0 to 1)

## Example

```json
{
  "text": "moi maailma",
  "language": "fi",
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": "moi maailma",
      "avg_logprob": -0.15,
      "no_speech_prob": 0.001,
      "words": [
        {"start": 0.0, "end": 0.4, "text": " moi", "probability": 0.99},
        {"start": 0.5, "end": 1.5, "text": " maailma", "probability": 0.98}
      ]
    }
  ]
}
```

(The numbers are illustrative.)

## Confidence

`avg_logprob` and `no_speech_prob` are segment-level values from faster-whisper. The service passes them through unchanged and defines no confidence scale, range or threshold for them:

- `avg_logprob` is the average log-probability of the segment's tokens, so it is at most 0; a value closer to 0 means the model was more certain.
- `no_speech_prob` is the model's estimate of the probability that the segment is not speech.
- Word `probability` is faster-whisper's probability for that word.

A consumer that needs a single confidence number has to derive it and pick its own rules. As an example, saarnavideo's current transcription worker (`transcription/transcribe.py` in the saarnavideo repository) sets its segment `confidence` to `clamp(avg_logprob + 1, 0, 1)`. That is what that worker does today; it is an example, not a recommendation of this service.

## API options

The `/inference` endpoint accepts these optional form fields besides `file`:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `language` | string | service default (`AUDITOR_STT_DEFAULT_LANGUAGE`, `fi`) | Language code |
| `model` | string | — | Accepted for whisper.cpp compatibility and ignored; the service runs one model at a time |
| `prompt` | string | none | Initial prompt for the decoder (for example liturgical terms) |
| `vad` | bool | false | Turns on faster-whisper's VAD filter, which removes non-speech parts before decoding |
| `temperature` | float | faster-whisper's default | Sampling temperature, passed on as given (the service does not restrict the range) |
| `response_format` | string | `json` | `json`, `verbose_json` or `text` |

Example:

```bash
curl -X POST http://localhost:8090/inference \
  -F "file=@segment.wav" \
  -F "language=fi" \
  -F "prompt=raamattu kristillinen liturgia" \
  -F "vad=true" \
  -F "temperature=0.5"
```

The `/v1/audio/transcriptions` (OpenAI-compatible) endpoint accepts `file`, `model` (ignored), `language`, `prompt`, `temperature` and `response_format`, but not `vad`.

Options that are not sent are not passed to faster-whisper, so its own defaults apply.

## Batch jobs

The jobs API (`/v1/jobs`) returns word-level timing and the segment fields above; with `word_timestamps=false` the `words` lists are empty. Its submission form fields are `file` or `source_path`, `language`, `chunk_seconds`, `word_timestamps`, `prompt` and `client_ref`; there are no `vad` or `temperature` fields (the VAD filter inside each chunk is controlled by the server setting `AUDITOR_STT_BATCH_VAD`). Full reference: [Batch jobs API](batch-jobs.md).

| Field | Type | Default |
|-------|------|---------|
| `prompt` | string | none |
| `chunk_seconds` | float | `60.0` (range 5 to 300) |
| `word_timestamps` | bool | `true` |

The batch result has these additional fields:

- `complete`: `true` when every planned chunk is present in the result, `false` for a partial result (`partial=1` on a job that is still running or failed).
- `chunks_done`, `chunks_total`, `duration_seconds`.
- `models`: the distinct model ids that produced the chunks, in chunk order (more than one entry means the model was switched during the job).

## Possible uses

- Transcript navigation: jump to a moment in the audio with the word or segment timing.
- Captions: the jobs result route groups words into cues (VTT, SRT, YouTube) using their timing and punctuation.
- Filtering or weighting by the confidence values above, with rules chosen by the consumer.
- Alignment of the transcript with video using word and segment timestamps.
