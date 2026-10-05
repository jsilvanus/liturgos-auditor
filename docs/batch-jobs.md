# Batch Jobs API

The `/v1/jobs` API transcribes full audio and video files in checkpointed chunks. Every finished chunk is written to disk, so an abort or restart does not lose the work. Jobs are submitted by file upload, by a path under the server's media root, or by a URL the server fetches itself, run asynchronously (one job at a time, in submission order), and are polled for status and results.

The API needs a data directory on the server (`AUDITOR_STT_DATA_DIR`). Without one every `/v1/jobs` route answers `503`. `auditor-stt serve` and the Docker images set it by default; see the [README](../README.md#environment-variables).

The examples send `Authorization: Bearer YOUR_API_KEY`. That header is only needed when the service has `AUDITOR_STT_API_KEY` set; without a key the API is open.

## Submission

### Upload a file

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -F "file=@sermon.mp4" \
  -F "language=fi" \
  -F "chunk_seconds=60" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Use a path on the media volume

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -F "source_path=/media/sermons/sermon.mp4" \
  -F "language=fi" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Fetch a URL

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -F "source_url=https://storage.example.com/bucket/sermon.wav?X-Amz-Signature=..." \
  -F "language=fi" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

The service downloads the file into the job's directory before it answers, so the caller does not upload anything (a presigned S3 URL works). Only hosts listed in `AUDITOR_STT_SOURCE_URL_HOSTS` are fetched: comma-separated host names, `*` patterns allowed (for example `s3.example.com,*.amazonaws.com`). Without the setting `source_url` is rejected with 422. Only `http` and `https` URLs without credentials are accepted, redirects are not followed, and the size cap is the same as for uploads (`AUDITOR_STT_MAX_UPLOAD_MB`, 413). A URL the service cannot use, a host that is not listed and a failed fetch all answer 422 with a short message. Like an upload, the fetched copy is removed when the job ends.

### Stripping the audio on an fffleet worker

`source_url` may point at a full video. By default the service downloads it and extracts the audio with its own ffmpeg. With `AUDITOR_STT_STRIP=fleet` the service does not download the file at all: it submits an [fffleet](https://github.com/jsilvanus/fffleet) batch job (`ffmpeg -i <url> -vn -ac 1 -ar 16000 -c:a pcm_s16le`) whose input is the URL and whose output is an HTTP PUT back to the service. Only the 16 kHz mono WAV (about 115 MB per hour) reaches the service, and the machine running the model needs no ffmpeg and no bandwidth for the video.

- The fleet job id is `auditor-strip-<job id>`, so a service restart resubmits to the same fleet job. A restart after the audio arrived just carries on.
- The worker uploads to `PUT /v1/jobs/{id}/audio?token=...` on `AUDITOR_STT_PUBLIC_URL`. The token is generated per job, accepted for one upload while the job waits for its audio, and removed from the job's manifest once the audio has arrived (the source URL is removed too, as it may carry a signature). The upload is size-capped (`AUDITOR_STT_MAX_INGEST_MB`) and must be a 16-bit PCM mono WAV (422 otherwise). The route does not use the API key, because fleet workers do not hold one; the token is the authorisation.
- The URL and the host check stay as above: the service validates `source_url` against `AUDITOR_STT_SOURCE_URL_HOSTS` before the fleet sees it. Redirect behaviour on the worker side is the worker's (fffleet fetches inputs with its normal HTTP client).
- If the fleet cannot be reached, the service strips locally as before (fetch, then its own ffmpeg) unless `AUDITOR_STT_FLEET_FALLBACK=off`, in which case the job fails with "The fleet is unavailable". A fleet job that ran and failed (for example unreadable media) fails the job; the message never contains the URL.
- Uploads and `source_path` jobs are not affected: their file is already here.

## Submission parameters

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `file` | File | — | Multipart file upload; exactly one of `file`, `source_path` or `source_url` is required (otherwise 422) |
| `source_path` | string | — | Path of an existing file on the **server**, inside `AUDITOR_STT_MEDIA_ROOT`. An absolute path must lie under the media root; a relative path is resolved against it. Symlinks and `..` are resolved first and cannot lead out of the root. Anything else answers 422 with the same message, so it does not reveal whether a file exists elsewhere. 422 as well when `AUDITOR_STT_MEDIA_ROOT` is not set |
| `source_url` | string | — | `http(s)` URL the service fetches itself. The host must be listed in `AUDITOR_STT_SOURCE_URL_HOSTS`; unset: 422. See [Fetch a URL](#fetch-a-url) |
| `language` | string | service default language (`AUDITOR_STT_DEFAULT_LANGUAGE`, `fi`) | Passed to faster-whisper as given; the value is not validated at submission |
| `chunk_seconds` | float | `60.0` | Chunk size in seconds; allowed range `[5.0, 300.0]`, otherwise 422 |
| `word_timestamps` | bool | `true` | Include word-level timing. With `false`, segments have an empty `words` list and captions use one cue per segment |
| `prompt` | string | none | Initial prompt (for example liturgical vocabulary). The tail of the previous chunk's text is appended to it for each chunk (see [Chunking](#chunking)) |
| `client_ref` | string | none | Opaque identifier, at most 200 characters (422 if longer), for filtering in list requests, e.g. a project id |

## Submission response

Status `202 Accepted`:

```json
{
  "id": "3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d",
  "status": "queued"
}
```

The `Location` header holds the job URL: `/v1/jobs/{id}`. Job ids are 32 hexadecimal characters.

A job is accepted while the model is still loading. The runner normalises and plans the job, then waits for a loaded model before the first chunk (phase `waiting_for_model`) instead of the submission being rejected.

## Upload limits

`AUDITOR_STT_MAX_UPLOAD_MB` (default 2048; 1 MB here is 1024 x 1024 bytes) caps uploads to `POST /v1/jobs`. It is not applied to the live routes. FastAPI parses a multipart body before the route runs, so the `UploadGuard` middleware answers `POST /v1/jobs` first, before the body is read, and closes the connection when it refuses:

- **401**: an API key is configured and the `Authorization` header is missing or wrong.
- **503**: jobs are not configured (no `AUDITOR_STT_DATA_DIR`).
- **413**: the announced `Content-Length` exceeds the cap (plus 1 MiB of allowance for multipart framing), or a body sent without `Content-Length` exceeds it while streaming. The route itself then enforces the exact file size while saving the upload and also answers 413. A rejected upload leaves nothing on disk.

A client that is still sending the body when the server refuses it may see a connection reset instead of the status code (`scripts/video-to-vtt-jobs.py` runs a pre-flight check for this reason).

## Job status

Poll with `GET /v1/jobs/{id}`:

```json
{
  "id": "3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d",
  "status": "running",
  "phase": "transcribing",
  "error": null,
  "client_ref": "project-123",
  "created_at": "2024-09-21T12:34:56.123456+00:00",
  "started_at": "2024-09-21T12:34:57.234567+00:00",
  "finished_at": null,
  "progress": 45.2,
  "current_seconds": 1814.5,
  "total_seconds": 4020.0,
  "eta_seconds": 1543.0,
  "chunks_done": 30,
  "chunks_total": 67,
  "params": {
    "language": "fi",
    "chunk_seconds": 60.0,
    "word_timestamps": true
  },
  "cancel_requested": false
}
```

| Field | Notes |
|-------|-------|
| `status` | `queued`, `running`, `completed`, `failed` or `cancelled` |
| `phase` | Label of what the runner is doing: `queued`, `normalising`, `transcribing`, `waiting_for_model` (no model loaded yet), and finally the end status (`completed`, `failed`, `cancelled`) |
| `error` | `null` unless the job failed; then a short message (for unexpected errors the exception class and message, cut at 300 characters) |
| `created_at`, `started_at`, `finished_at` | ISO-8601 UTC. `started_at` is when the current run began (reset when a job is requeued after a restart); `null` until the job runs. `finished_at` is `null` until the job ends |
| `progress` | Percentage `[0, 100]`, one decimal, based on finished chunks |
| `current_seconds` | Audio position reached: the end of the last finished chunk |
| `total_seconds` | Duration of the normalised audio; `0` until the audio has been normalised |
| `eta_seconds` | Wall-clock time of the current run divided by the audio seconds finished in it, times the audio seconds left. `null` for failed and cancelled jobs and until a chunk has finished in the current run; `0` when all chunks are done |
| `chunks_done`, `chunks_total` | Chunk-level progress. `chunks_total` is `0` until the chunk plan exists |
| `params` | The `language`, `chunk_seconds` and `word_timestamps` the job runs with (`prompt` is not echoed) |
| `cancel_requested` | `true` once a `DELETE` has asked a running job to stop and it has not stopped yet |

## Results

Fetch with `GET /v1/jobs/{id}/result?format=…` (`json` is the default). Formats: `json`, `vtt`, `srt`, `text`, `youtube`; anything else is a 422.

Without `partial=1` the result is only available once the job has status `completed`; before that (and for failed or cancelled jobs) the answer is `409` with `{"detail": "Job is not finished", "status": "<current status>"}`.

### JSON format

```bash
curl "http://localhost:8090/v1/jobs/{id}/result?format=json" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Returns the assembled result with absolute timestamps (media time) and word timing:

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
  ],
  "complete": true,
  "chunks_done": 1,
  "chunks_total": 1,
  "duration_seconds": 1.5,
  "models": ["large-v3-turbo"]
}
```

(The numbers are illustrative.)

| Field | Notes |
|-------|-------|
| `text` | The chunk texts joined with single spaces |
| `language` | Language reported for the first chunk that has one, otherwise the requested language |
| `segments[].start`, `segments[].end` | Absolute timestamps in seconds |
| `segments[].avg_logprob`, `segments[].no_speech_prob` | Segment-level values from faster-whisper, passed through unchanged (may be `null`); see [Word-level transcription](word-level-transcription.md#confidence) |
| `segments[].words[]` | Word timing and probability. `text` is not trimmed and normally has a leading space |
| `complete` | `true` when the result of every planned chunk is present, `false` for a partial result |
| `chunks_done`, `chunks_total`, `duration_seconds` | Chunks included, chunks planned, duration of the normalised audio |
| `models` | Distinct model ids that produced the chunks, in chunk order. More than one entry means the model was switched during the job. Absent if no chunk recorded a model |

### VTT, SRT, and text formats

```bash
curl "http://localhost:8090/v1/jobs/{id}/result?format=vtt" \
  -H "Authorization: Bearer YOUR_API_KEY"

curl "http://localhost:8090/v1/jobs/{id}/result?format=srt" \
  -H "Authorization: Bearer YOUR_API_KEY"

curl "http://localhost:8090/v1/jobs/{id}/result?format=text" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

| `format` | Media type |
|----------|------------|
| `json` | `application/json` |
| `vtt` | `text/vtt; charset=utf-8` |
| `srt` | `application/x-subrip; charset=utf-8` |
| `text`, `youtube` | `text/plain; charset=utf-8` |

Every non-JSON response carries an `X-Job-Complete: true` or `false` header (the JSON body has `complete`). `text` is the whole transcript on one line.

### YouTube Live Captions format

Requires `start_time`: the wall-clock time of media t=0, ISO-8601. A timestamp without an offset is taken as UTC; an offset is allowed but should be URL-encoded (`%2B02:00`). Missing or unparsable `start_time` is a 422.

```bash
curl "http://localhost:8090/v1/jobs/{id}/result?format=youtube&start_time=2024-09-21T12:34:56Z" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

One record per cue: a timestamp line (absolute UTC, `YYYY-MM-DDTHH:MM:SS.mmm`, no offset), then the cue text on a single line. Records end with a newline. This is the format lcyt's sender uses. Illustrative output:

```
2024-09-21T12:34:56.000
moi maailma
2024-09-21T12:34:57.500
...
```

Optional query parameters:
- `region`: region identifier
- `cue`: cue identifier

When neither is given, no suffix is written. When at least one is given, the timestamp line gets ` region:<region>#<cue>` and a missing one falls back to `reg1` / `cue1`. Whitespace in either value is a 422.

```bash
curl "http://localhost:8090/v1/jobs/{id}/result?format=youtube&start_time=2024-09-21T12:34:56Z&region=reg1&cue=cue1" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Illustrative output:

```
2024-09-21T12:34:56.000 region:reg1#cue1
moi maailma
...
```

### Partial results

Add `partial=1` to get the finished chunks of a job that is still running, or of a failed or cancelled job:

```bash
curl "http://localhost:8090/v1/jobs/{id}/result?format=json&partial=1" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

The response has `"complete": false` (or `true` if every chunk happens to be present) and `chunks_done` / `chunks_total`. For a job that has not finished a chunk yet the result is empty.

### Cue customization

VTT, SRT and YouTube output group words into cues. Query parameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_cue_duration` | `7.0` | Maximum cue duration in seconds; must be greater than 0 (else 422) |
| `max_line_chars` | `42` | Target maximum characters per line; at least 1 (else 422) |

A cue ends when it would exceed `max_cue_duration`, when it gets wider than twice `max_line_chars` (characters without spaces), or after a word ending in `.`, `!`, `?`, `:` or `;`. A cue has at most two lines; the second line may be longer than `max_line_chars`. Segments without word timing (`word_timestamps=false`) become one cue each.

## List and filter jobs

```bash
curl http://localhost:8090/v1/jobs \
  -H "Authorization: Bearer YOUR_API_KEY"

curl "http://localhost:8090/v1/jobs?client_ref=project-123" \
  -H "Authorization: Bearer YOUR_API_KEY"

curl "http://localhost:8090/v1/jobs?status=completed" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Response: all matching jobs, newest first (no paging). An unknown `status` value is a 422.

```json
{
  "jobs": [
    { ... job status ... }
  ]
}
```

## Cancel and delete

```bash
curl -X DELETE http://localhost:8090/v1/jobs/{id} \
  -H "Authorization: Bearer YOUR_API_KEY"
```

- **Running job**: returns `202` with `{"id": "...", "status": "running", "cancel_requested": true}`. The runner stops between chunks (a chunk that is being transcribed is finished first; ffmpeg is interrupted if the job is still normalising) and then deletes the whole job. Afterwards `GET /v1/jobs/{id}` answers 404.
- **Queued, completed, failed or cancelled job**: returns `204`; the whole job directory is deleted at once.
- **Unknown id**: returns `404`.
- **`409` "Job is busy; try again"**: the files could not be deleted right now (typically on Windows, when the runner still has the job audio open). Retry after a moment.

## How it works

### Chunking

1. **Normalisation**: ffmpeg decodes the input once to a 16 kHz mono 16-bit WAV in the job directory (only local files are read). Chunks are then read from that WAV one at a time, so the whole file is never held in memory.
2. **Planning**: chunks are placed on a nominal grid (`k × chunk_seconds`). Each internal cut is moved to the middle of the longest non-speech gap (at least 0.3 s, found with the Silero VAD in faster-whisper) within ±5 seconds of the grid point; the window is a quarter of `chunk_seconds` when that is smaller. If the VAD fails, the quietest 100 ms frame in the window is used. If no gap is found the cut stays on the grid and is marked as not snapped in `manifest.json`. A final piece shorter than a quarter of `chunk_seconds` is merged into the previous chunk. The plan is stored in the manifest and never recomputed on resume.
3. **Transcription**: chunks are transcribed in order at batch priority. Live requests are served before the next chunk starts; a chunk already running is not interrupted. Each chunk is decoded independently (`condition_on_previous_text` off). The last `AUDITOR_STT_CARRY_CONTEXT_CHARS` characters (default 200, starting at a word boundary; `0` turns it off) of the previous chunk's text are appended to the job's `prompt` as context. The VAD filter runs inside each chunk unless `AUDITOR_STT_BATCH_VAD` is false. A failed chunk is retried twice (waiting 1 s, then 4 s) before the job fails.
4. **Persistence**: each chunk result is written to `chunks/00042.json` (five-digit index) via a `.tmp` file that is renamed into place. A chunk file that exists is a finished chunk.

### Recovery

At startup the runner scans `<AUDITOR_STT_DATA_DIR>/jobs/*`:
- Jobs with status `running` (interrupted by a crash or a stop) go back to `queued`; all queued jobs are resubmitted, oldest first.
- Uploaded copies and normalised audio still lying around for finished jobs are removed.
- When a job runs again, chunks whose result file exists are skipped and the stored plan is reused. The normalised audio is reused if it is still there, otherwise the source is normalised again.

An abort, crash or restart loses at most the chunk that was in flight. The cancel flag is checked between chunks.

### Lifetime and what is deleted

| When | Deleted | Kept |
|------|---------|------|
| A job ends (completed, failed or cancelled) | The uploaded copy (`source/`) and the normalised `audio.wav` | `manifest.json` and `chunks/*.json` (the transcript) |
| `AUDITOR_STT_JOB_TTL_HOURS` (default 72) after the job ended | The whole job directory | — |
| `DELETE` | The whole job directory (for a running job: once the runner has stopped) | — |
| An upload that fails or is refused while being received | The whole job directory | — |

The TTL counts from `finished_at`. A sweep runs at startup and then once an hour, so a job can outlive its TTL by up to about an hour. Queued and running jobs are never purged. A file submitted by `source_path` lives on the media volume and is never modified or deleted.

## Error codes

- **202**: job accepted and queued; also the answer to `DELETE` on a running job.
- **204**: job deleted.
- **401**: an API key is configured and the header is missing or wrong.
- **404**: unknown job id (also for a malformed id, and for `DELETE` of an unknown job).
- **409**: `GET .../result` on a job that is not `completed` without `partial=1` (body `{"detail": "Job is not finished", "status": ...}`); `DELETE` when the files are busy ("Job is busy; try again").
- **413**: upload larger than `AUDITOR_STT_MAX_UPLOAD_MB`.
- **422**: invalid input: not exactly one of `file` / `source_path` / `source_url`; a `source_url` that is not enabled, not allowed, unusable or failed to download; empty upload; `chunk_seconds` outside 5-300; `client_ref` longer than 200 characters; `source_path` not an existing file inside the media root, or no media root configured; unknown `format` or `status`; `format=youtube` without a valid `start_time`, or whitespace in `region` / `cue`; `max_cue_duration` not above 0 or `max_line_chars` below 1.
- **500**: a complete result was requested but chunk result files of the job are missing or unreadable ("Result files of this job are missing").
- **503**: jobs are not configured (no `AUDITOR_STT_DATA_DIR`). A model that is still loading does not cause 503 on the jobs routes.

## Failure and resume

A **failed job**:
- Has `status: "failed"` and an `error` message. Examples: `Chunk 3 failed after 3 attempts: RuntimeError: ...`, `Could not decode audio from <file>: ...` (ffmpeg could not read the input), `The media contains no audio`.
- Keeps the chunks finished before the error. Fetch them with `GET /v1/jobs/{id}/result?partial=1`; without `partial=1` the result route answers 409.
- Has already lost its uploaded copy and normalised audio, and is purged after the TTL like any finished job.

A **cancelled job**: `DELETE` on a running job ends in deletion (see above). The runner marks the job `cancelled` just before it deletes it, so in practice the job disappears (404) rather than staying visible as `cancelled`. If that deletion fails, the job stays as `cancelled` with its partial chunks until the TTL.

An **interrupted job** (process crash, out of memory, or a stop of the service):
- Keeps the chunks finished so far.
- Is `running` in its manifest until the service starts again; then it is requeued and continues with the unfinished chunks.
- Needs nothing from the client: keep polling the same id. While the service is down the requests simply fail (`scripts/video-to-vtt-jobs.py` retries them; see [video-to-vtt-jobs](video-to-vtt-jobs.md)).

## Performance notes (CPU-only deployments)

Preemption happens only between chunks, never inside one. On a CPU-only deployment a live request can therefore wait for the batch chunk that is running. Mitigate by:
- Reducing `chunk_seconds` (for example to 30), so live requests get a turn more often.
- Running a separate service instance for batch jobs.

No transcription speed is measured or promised in this repository; see the README's architecture notes for the estimate it cites.
