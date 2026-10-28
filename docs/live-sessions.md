# Live sessions (pull type)

The service pulls a live stream itself and publishes the text as Server-Sent Events. The client does not decode or chunk audio; it starts a session, listens, and stops it.

```
POST   /v1/live               {"source": "rtsp://mediamtx:8554/live/main", "language": "fi", "prompt": "...", "client_ref": "...", "delay_seconds": 0}  -> 202
GET    /v1/live               sessions
GET    /v1/live/{id}          status
GET    /v1/live/{id}/events   text/event-stream; Last-Event-ID resumes within the last 500 events
DELETE /v1/live/{id}          stop (cancels the fleet job)
```

All routes need the API key (when one is set). `source` must be `rtsp://`, `rtsps://` or `srt://` and its host must match `AUDITOR_STT_LIVE_SOURCE_HOSTS`; without that setting the feature is off (503). More than `AUDITOR_STT_MAX_LIVE_SESSIONS` active sessions answer 429.

## Optional PCM pre-roll

Existing `application/json` start requests remain unchanged. To submit buffered audio before the pulled stream, send `multipart/form-data` with a `metadata` field containing the usual start JSON and a binary `preroll` part:

```bash
curl -X POST http://localhost:8090/v1/live \\
  -F 'metadata={"source":"rtsp://mediamtx:8554/live/main","language":"fi","client_ref":"pivoice-main","has_preroll":true}' \\
  -F 'preroll=@preroll.pcm;type=application/octet-stream'
```

The pre-roll must be raw **16 kHz mono signed 16-bit little-endian PCM (S16LE)**, not a WAV file. Its duration is inferred from its byte length (32,000 bytes per second); there is no duration field. `has_preroll` defaults to false. If true, a non-empty binary part is required. If false, an unexpected non-empty pre-roll part is rejected. JSON requests may omit `has_preroll` or set it to false; true requires multipart because it needs the binary payload.

For safety, the pre-roll is limited to 10 MiB (about 5 minutes of PCM). The live source is opened and read immediately into a bounded FIFO (up to 32 seconds of PCM) while pre-roll is segmented. Auditor then feeds pre-roll through the same segmenter before draining the FIFO and continuing with live PCM, so audio is processed in order and only once. If the FIFO overflows, a `status` event with `state: "dropped"` and `reason: "live_buffer_overflow"` is emitted rather than silently hiding the loss. Because raw PCM has no capture timestamps, the service anchors the end of pre-roll near live-source startup; startup and transport latency can affect absolute wall-clock alignment.

## Events

| event | data |
|---|---|
| `transcript` | `sequence`, `text`, `language`, `start`, `end` (seconds since the audio timeline begins; with pre-roll, from the start of the pre-roll), `wall_start`, `wall_end` (UTC, ISO 8601 with milliseconds), `delay_seconds` |
| `status` | `state`: `starting`, `running`, `reconnecting`, `clock_reset` (`drift_seconds`), `dropped` (a segment skipped because the model was busy or behind), `ended` (`reason`: `stopped`, `source_lost`, `error`, `shutdown`) |
| `error` | `message`, with URLs and credentials removed |

Only final transcripts are sent, one per speech segment (cut at pauses of 0.6 s, or in the longest gap once a segment reaches 10 s). Each carries its own absolute time, so a consumer never needs to know when the session started. The `sequence` keeps counting across reconnects.

## Delay

`delay_seconds` (request field; default `AUDITOR_STT_LIVE_DELAY_SECONDS`, `0`; at most `AUDITOR_STT_LIVE_MAX_DELAY_SECONDS`, `120`, else 422) holds each transcript back until `wall_end + delay_seconds`, so the text lines up with a stream that is itself delayed, for example a broadcast delay on the way to viewers. Only the moment of publishing moves: `start`, `end`, `wall_start` and `wall_end` stay the times of the source, and the event carries `delay_seconds` so a consumer can tell. Order and `sequence` are kept, and held text does not slow transcription of the next segments. A transcript already older than its release time is published at once. When the stream ends, the session stays open until the held text has been published; `DELETE` ends it immediately and drops what is still held. `status` events (`dropped`, `clock_reset`) are not delayed. The delay is measured against this host's clock, like the wall times.

## How it runs

1. With `AUDITOR_STT_STRIP=fleet` the service submits an fffleet **stream** job (`stdout: true`): `ffmpeg -rtsp_transport tcp -i <source> -vn -ac 1 -ar 16000 -f s16le pipe:1`, with `requires` from `AUDITOR_STT_LIVE_REQUIRES`. Use the generic `net:` capability so the job only lands on a worker that can reach the stream: start workers with `FFFLEET_PROBE=mediamtx=10.1.2.3:8554` and set `AUDITOR_STT_LIVE_REQUIRES=net:mediamtx`. The service reads the PCM from `GET /v1/jobs/{id}/stdout`. If the fleet is unreachable the same ffmpeg runs in this process, unless `AUDITOR_STT_FLEET_FALLBACK=off`. Without `AUDITOR_STT_STRIP=fleet` it always runs locally.
2. A reader thread cuts the PCM into segments (Silero VAD, loudness as fallback). Segments go through the inference queue at live priority, so they overtake batch chunks. While a session runs, `GET /status` reports `live.active` and queue estimates for batch jobs are marked as minimums. The previous text is passed as prompt.
3. When the stream ends or breaks, the session reconnects with backoff (1 s up to 10 s) for `AUDITOR_STT_LIVE_RECONNECT_SECONDS` without receiving data, then ends with `source_lost`. Every attempt is its own fleet job, `auditor-live-<session>-<attempt>`.

## Wall time

The PCM has no clock. The first block read pins an anchor (wall time of its end); every later time is the anchor plus the sample position. While the reader has no backlog it compares the stream position with the wall clock, and when they differ by more than `AUDITOR_STT_LIVE_CLOCK_DRIFT_SECONDS` it moves the anchor and sends `clock_reset`. A backlog is old audio and does not move it. Times are as good as the host clocks (use NTP); the latency of the pipe from the worker is not subtracted, expect a fraction of a second to a few seconds of lateness.

## Not durable

Sessions live in memory. A restart ends them (the SSE connection closes); the client opens a new one. Ended sessions stay readable for five minutes. A stream job left on the fleet by a crash ends when its reader goes away.

## Not verified

Everything is tested against a fake fleet and synthetic audio. It has not run against a real MediaMTX, fleet or GPU.
