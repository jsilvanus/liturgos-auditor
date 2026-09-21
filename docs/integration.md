# Integration guide

This document describes how each consumer talks to the `liturgos-auditor-stt` service. The consumer-side facts come from the adapters in `live-captions-yt` (`packages/plugins/lcyt-rtmp/src/stt-adapters/whisper-http.js` and `openai.js`) and from the `saarnavideo` repository.

Authentication in one sentence: when the service has `AUDITOR_STT_API_KEY` set, every endpoint except `/health` requires `Authorization: Bearer <key>` (401 otherwise); when it has none, every endpoint is open. See the [README](../README.md#all-endpoints).

## lcyt (live-captions-yt)

lcyt posts short audio pieces to transcribe live stream captions. It sends either an fMP4 HLS segment (`segment.mp4`, `audio/mp4`) or, on its PCM path, a WAV file (`segment.wav`, 16 kHz mono; the PCM buffer flushes after at least 2 s and at most 10 s of audio by default). lcyt selects the STT provider in its settings: `whisper_http` or `openai` (both described below).

### Using the Whisper HTTP adapter (no auth)

lcyt's `WhisperHttpAdapter` (provider `whisper_http`) posts to `/inference`. Its base URL comes from the `stt.whisper_http_url` setting or the `WHISPER_HTTP_URL` environment variable (required), for example `http://localhost:8090`; `WHISPER_HTTP_MODEL` is optional.

```bash
curl -X POST http://localhost:8090/inference \
  -F "file=@segment.mp4" \
  -F "language=fi"
```

**Request fields sent by the adapter:**
- `file`: the audio (multipart)
- `language`: short ISO 639-1 code (for example `fi`; lcyt cuts a BCP-47 code down to its first part)
- `model`: only if configured in lcyt; ignored by the service, which runs one model at a time

**Other fields the service accepts** (lcyt does not send them): `prompt`, `vad`, `temperature`, `response_format`; see [Word-level transcription](word-level-transcription.md).

**Response** (`response_format` `json`, the default; `verbose_json` returns the same):

```json
{
  "text": "tuore tieto",
  "language": "fi",
  "segments": [
    {
      "start": 0.0,
      "end": 1.2,
      "text": "tuore tieto",
      "avg_logprob": -0.2,
      "no_speech_prob": 0.01,
      "words": [
        {"start": 0.0, "end": 0.5, "text": " tuore", "probability": 0.98},
        {"start": 0.6, "end": 1.2, "text": " tieto", "probability": 0.99}
      ]
    }
  ]
}
```

(The numbers are illustrative.) lcyt reads only the top-level `text` field. An empty `text` is dropped by the adapter without an event, so that chunk produces no caption.

**Timeout and errors (adapter behaviour):**
- 60 second timeout per request, no retry.
- The adapter never sends an `Authorization` header, so this route only works against a service without `AUDITOR_STT_API_KEY` (network isolation is the protection in that setup).
- A network error, a non-2xx response or a body that is not JSON makes the adapter emit an `error` event (the STT manager re-emits it); the chunk is lost.

### Using the OpenAI adapter (with auth)

lcyt's `OpenAiAdapter` (provider `openai`) posts to `<base URL>/v1/audio/transcriptions`, so it can use the service's OpenAI-compatible route. Set on the lcyt side (settings `stt.openai_stt_url`, `stt.openai_stt_api_key`, `stt.openai_stt_model`, or the environment variables):

- `OPENAI_STT_URL=http://localhost:8090` (default: `https://api.openai.com`)
- `OPENAI_STT_API_KEY=YOUR_KEY` (required: the adapter refuses to start without one)
- `OPENAI_STT_MODEL` (default `whisper-1`; the service ignores it)

On the service side, set `AUDITOR_STT_API_KEY=YOUR_KEY` to make the route require that key. If the service has no key configured it accepts any `Authorization` header.

The adapter sends `Authorization: Bearer <key>` and the fields `file`, `model`, `language` and `response_format=json`. It never sends `prompt` or `temperature`. Equivalent request:

```bash
curl -X POST http://localhost:8090/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_KEY" \
  -F "file=@segment.mp4" \
  -F "model=whisper-1" \
  -F "language=fi" \
  -F "response_format=json"
```

Fields the service accepts:
- `file`: audio (required)
- `model`: ignored; the service runs one model at a time
- `language`: language code; the service default (`AUDITOR_STT_DEFAULT_LANGUAGE`, `fi`) when omitted
- `response_format`: `json` (default), `verbose_json` or `text`; anything else is a 400
- `prompt`: optional vocabulary hint (an extension; the service supports it)
- `temperature`: optional sampling temperature, passed to faster-whisper as given

Response for `json`:

```json
{
  "text": "tuore tieto"
}
```

`verbose_json` returns the service's own full result (`text`, `language`, `segments` with words), not OpenAI's exact `verbose_json` shape; `text` returns a plain-text body. lcyt requests `json` and reads only `text`.

### Session overlap and the queue

Requests from one lcyt session can overlap (for example when transcription takes longer than the segment interval). The service runs one inference call at a time. Live requests go before batch job chunks and are served in FIFO order among themselves. At most `AUDITOR_STT_MAX_QUEUE` live requests (default 8, counting the one running) are in the queue; the next one is rejected at once with `503` and lcyt gets the error event described above. A live request also waits for a batch chunk that is already running (see [Batch jobs](batch-jobs.md#performance-notes-cpu-only-deployments)).

### Failure handling

| Response | Meaning |
|----------|---------|
| `200` with `"text": ""` | Nothing was recognised (for example silence). On `/inference` the body is `{"text": "", "language": "fi", "segments": []}` |
| `400` | Unsupported `response_format` |
| `401` | An API key is configured and the header is missing or wrong |
| `422` `{"detail": "Could not decode audio"}` | The bytes could not be decoded as audio; lcyt treats it as an error |
| `503` `Model not loaded yet` | The model is still loading or failed to load (`/health` also answers 503) |
| `503` `Inference queue depth exceeded (N)` | The live queue is full |

**Not verified in this repository:** whether headerless fMP4 HLS segments (as output by MediaMTX or ffmpeg with `-movflags frag_keyframe+empty_moov`) decode standalone in faster-whisper. No test in this repository covers it.

## saarnavideo

The jobs API (`/v1/jobs`) is written for saarnavideo's full-file transcription (the service code says so). In the saarnavideo checkout this was written against there is no client for it yet: its transcription today is `transcription/transcribe.py`, a local faster-whisper worker started by `PythonTranscriptionProvider`, behind the `TranscriptionProvider` interface (`transcribe(inputPath, language?)`) in `src/domain/transcription.ts`. The mappings below show how this service's job status and result line up with saarnavideo's `MediaJob` progress fields and transcript schema; they are what a caller of the jobs API would do, not existing saarnavideo code.

### Submit a job

Submit with the path of the file on the shared media volume (no upload needed):

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -F "source_path=/media/sermons/service-2024-09-21.mp4" \
  -F "language=fi" \
  -F "chunk_seconds=60" \
  -F "word_timestamps=true" \
  -H "Authorization: Bearer YOUR_KEY"
```

The service needs:
- `AUDITOR_STT_DATA_DIR` set (enables jobs; the Docker images and `auditor-stt serve` set it).
- `AUDITOR_STT_MEDIA_ROOT` pointing at the shared media directory, for `source_path` submissions. `source_path` is a path on the service's side, inside that directory.
- `AUDITOR_STT_API_KEY` if requests should be authenticated (the header is then required).

Response (`202`, with a `Location: /v1/jobs/{id}` header):

```json
{
  "id": "3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d",
  "status": "queued"
}
```

### Poll for status

```bash
curl http://localhost:8090/v1/jobs/{id} \
  -H "Authorization: Bearer YOUR_KEY"
```

saarnavideo's `MediaJob` (`prisma/schema.prisma`) has these progress fields:

| Service | saarnavideo `MediaJob` |
|---------|------------------------|
| `status` (`queued`, `running`, `completed`, `failed`, `cancelled`) | `status` (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`): the same five names |
| `progress` (percentage, one decimal) | `progress` (`Int`): round it |
| `phase` (`queued`, `normalising`, `transcribing`, `waiting_for_model`, then the end status) | `phase` (`String?`) |
| `current_seconds` | `currentMs` (`BigInt?`): `seconds * 1000` |
| `total_seconds` | `totalMs` (`BigInt?`): `seconds * 1000` |
| `eta_seconds` (may be `null`) | `etaSeconds` (`Int?`): round it |
| `error` | `error` (`String?`) |

`chunks_done` / `chunks_total` and `cancel_requested` have no counterpart in `MediaJob`. Field meanings: [Job status](batch-jobs.md#job-status).

### Fetch the result

```bash
curl "http://localhost:8090/v1/jobs/{id}/result?format=json" \
  -H "Authorization: Bearer YOUR_KEY"
```

saarnavideo's transcript schema (`src/domain/transcription.ts`) is `{version: 1, language, segments[]}` with `segments[]` = `{startSeconds >= 0, endSeconds > 0 and > startSeconds, text, confidence? in 0..1}`. Mapping:

| Service result | saarnavideo |
|---|---|
| `segments[].start` | `startSeconds` |
| `segments[].end` | `endSeconds` |
| `segments[].text` | `text` |
| `segments[].avg_logprob` | `confidence` (optional): derived by the consumer, see below |
| `language` | `language` |

The service passes faster-whisper's segment times through unchanged and does not check the `endSeconds > startSeconds` rule, so a consumer that validates against that schema should handle segments that violate it.

**Segment fields** (for advanced use):
- `segments[].avg_logprob`: faster-whisper's average log-probability of the segment. It is at most 0 and the service defines no scale or threshold for it; the consumer chooses how to turn it into a confidence (see [Word-level transcription](word-level-transcription.md#confidence)). Example: saarnavideo's current worker (`transcription/transcribe.py`) uses `clamp(avg_logprob + 1, 0, 1)`. That is what that worker does today, not a recommendation.
- `segments[].no_speech_prob`: faster-whisper's no-speech probability for the segment.
- `segments[].words[]`: word timing (`start`, `end` in seconds), `text` (untrimmed, normally with a leading space) and `probability`. Present unless the job was submitted with `word_timestamps=false`.

Example mapping:

```python
def compute_confidence(avg_logprob):
    # Example only: the formula saarnavideo's transcribe.py uses today.
    if avg_logprob is None:
        return None
    return max(0.0, min(1.0, avg_logprob + 1.0))


result = fetch_result(job_id)  # GET /v1/jobs/{id}/result?format=json

segments = []
for seg in result["segments"]:
    if seg["end"] <= seg["start"]:
        continue  # would fail saarnavideo's schema
    item = {"startSeconds": seg["start"], "endSeconds": seg["end"], "text": seg["text"]}
    confidence = compute_confidence(seg.get("avg_logprob"))
    if confidence is not None:
        item["confidence"] = confidence
    segments.append(item)

transcript = {"version": 1, "language": result["language"], "segments": segments}
```

### Cancel or delete

```bash
curl -X DELETE http://localhost:8090/v1/jobs/{id} \
  -H "Authorization: Bearer YOUR_KEY"
```

- Running job: returns `202`. Processing stops between chunks and the job is then deleted.
- Otherwise: returns `204` and the job is deleted. An unknown id returns `404`.

### Partial results during transcription

```bash
curl "http://localhost:8090/v1/jobs/{id}/result?format=json&partial=1" \
  -H "Authorization: Bearer YOUR_KEY"
```

Returns the finished chunks even if the job is still running (or failed). The response includes `"complete": false` unless every chunk is present. Without `partial=1` the result route answers `409` until the job is `completed`.

Uses: show transcript progress before the job completes; recover the finished part of a failed job.

### Error handling

| Status | Meaning |
|--------|---------|
| `401` | An API key is configured and the header is missing or wrong |
| `404` | Unknown job id |
| `409` | `GET .../result` before the job is `completed` and without `partial=1` (`{"detail": "Job is not finished", "status": ...}`); `DELETE` when the job's files are busy (retry) |
| `413` | Upload (if used) exceeds `AUDITOR_STT_MAX_UPLOAD_MB` |
| `422` | Invalid parameter (for example `source_path` not inside the media root, `chunk_seconds` outside 5-300). Fix and resubmit |
| `500` | A complete result was requested but chunk result files are missing ("Result files of this job are missing"). Check the service logs |
| `503` | Jobs are not configured (no `AUDITOR_STT_DATA_DIR`) |

A job that fails during processing is reported through `status: "failed"` and `error` on the status route, not through an HTTP error. Full list: [Batch jobs API](batch-jobs.md#error-codes).

### Alternative: upload instead of source_path

If the file is not on the shared media volume:

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -F "file=@sermon.mp4" \
  -F "language=fi" \
  -H "Authorization: Bearer YOUR_KEY"
```

The service stores the upload in the job directory and deletes that copy (and the normalised audio) as soon as the job ends. The transcript chunks stay until `AUDITOR_STT_JOB_TTL_HOURS` after the end or until `DELETE`.
