# Plan: optional PCM pre-roll for live sessions

## Goal

Extend the existing `POST /v1/live` start request so PiVoice can provide the PCM it has buffered before publishing its live stream. Auditor must process that pre-roll before the pulled MediaMTX stream, exactly once and in order, without changing existing JSON-only callers.

This is a plan only; implementation and tests have not been changed by this commit.

## Request contract

Keep the existing endpoint and response shape.

### Existing request remains supported

`application/json` continues to accept the current `StartRequest` fields and starts a session with no pre-roll. Omitting pre-roll must preserve current behaviour.

### New multipart request

Use `multipart/form-data` with:

- `metadata`: a JSON string containing the existing start fields plus optional `has_preroll`.
- `preroll`: optional raw PCM bytes, sent as an `application/octet-stream` part when `has_preroll` is true.

Example metadata:

```json
{
  "source": "rtsp://mediamtx:8554/live/main",
  "language": "fi",
  "prompt": "...",
  "client_ref": "pivoice-main",
  "delay_seconds": 0,
  "has_preroll": true
}
```

PCM format is fixed for the first version to match the live reader: **16 kHz, mono, signed 16-bit little-endian (S16LE)**. At this format the payload represents 32,000 bytes per second; duration is inferred from byte length, not requested in JSON and not configured separately.

### Flag and validation semantics

- `has_preroll` defaults to `false` when omitted.
- If false or omitted, the request follows the legacy no-pre-roll path; reject an unexpected non-empty `preroll` part rather than silently ignoring it.
- If true, require a non-empty `preroll` part.
- Reject malformed metadata, missing/invalid required `source`, a non-binary or empty pre-roll part when required, and a payload whose byte length is not divisible by two.
- Apply a bounded request/payload size limit to protect the service from unbounded uploads. This is a safety ceiling, **not** a target pre-roll duration; infer duration from the actual payload. Select the cap consistently with the service's existing request limits during implementation.
- Return clear 4xx validation errors. Do not create a session until request metadata and the pre-roll payload have passed validation.

## Processing and ordering

1. Validate the source using the existing scheme and host allow-list checks, and validate the existing delay/session limits as today.
2. Create the session with the optional pre-roll bytes attached to it.
3. Start pulling the RTSP/SRT source promptly. While the pre-roll is being consumed, buffer incoming live PCM in a bounded FIFO so live audio is not lost while pre-roll is segmented/transcribed.
4. Feed the pre-roll through the same PCM segmenter used by live audio, then drain buffered live PCM, then continue with newly read live PCM. Never run pre-roll through a separate inference path.
5. Ensure every sample is fed once, in order. If the bounded live buffer overflows, expose an explicit dropped/backlog status or error; do not silently reorder audio.
6. Keep the existing reconnect, cancellation, inference queue, SSE event, and transcript delivery behaviour intact.

### Timeline requirement

The current live clock anchors wall time to the first PCM block read. Pre-roll changes what “first block” means, so do not simply prepend bytes and leave the clock logic untouched. Define and test one consistent timeline before implementation:

- pre-roll is earlier audio and must precede the pulled stream in the session's sample timeline;
- transcript `start`/`end` and `wall_start`/`wall_end` must remain monotonic and meaningful;
- the live stream must not be accidentally counted twice or create a hidden gap at the pre-roll/live boundary.

Because PCM contains no embedded timestamps, exact capture-time alignment cannot be recovered from bytes alone. Document the chosen anchor convention and its accuracy. Prefer treating the end of the submitted pre-roll as the boundary immediately before the live PCM timeline, while accounting for stream startup/transport delay explicitly where the current source clock supports it.

## Implementation areas

- `python-packages/liturgos-auditor-stt/auditor_stt/serve/live/api.py`
  - Parse either legacy `application/json` or the new `multipart/form-data` request.
  - Keep the existing fields and response contract; add `has_preroll` only to the multipart metadata schema.
  - Validate all parts before creating a session.
- `python-packages/liturgos-auditor-stt/auditor_stt/serve/live/session.py`
  - Accept optional pre-roll PCM and coordinate pre-roll processing with live-source startup and bounded FIFO buffering.
  - Preserve cancellation/reconnect cleanup and the session's sample/wall-clock semantics.
- `python-packages/liturgos-auditor-stt/auditor_stt/serve/live/segmenter.py` and `clock.py`
  - Change only if needed to feed initial PCM through the existing segmentation path and anchor timestamps correctly; avoid duplicating VAD/segmentation logic.
- Tests in `python-packages/liturgos-auditor-stt/tests/test_live.py` (and focused session/segmenter tests if present).
- `docs/live-sessions.md`, README API documentation, and relevant request examples.

## Test plan

1. Existing JSON start request still creates a session and behaves exactly as before.
2. Multipart request with `has_preroll=false` starts a normal session.
3. Multipart request with valid pre-roll starts a session and processes pre-roll before live PCM.
4. `has_preroll=true` without a payload, empty payload, malformed metadata, wrong content type, and odd-byte PCM are rejected before session creation.
5. Unexpected pre-roll when the flag is false is rejected.
6. Test variable payload lengths and confirm inferred duration/sample count is correct.
7. Feed synthetic pre-roll and live PCM with recognizable segment boundaries; assert ordering, exactly-once delivery, and no loss at the boundary.
8. Test live PCM buffering while pre-roll is consumed, buffer overflow handling, cancellation, source reconnect, and shutdown.
9. Verify relative and wall-clock timestamps around the boundary, including that the first pre-roll segment is not incorrectly timestamped as newly captured live audio.
10. Run the existing live-session suite with fake sources/fleet only. Do not require real MediaMTX, fffleet workers, or a GPU in unit tests.

## Acceptance criteria

- Existing JSON clients require no changes.
- PiVoice can submit variable-length pre-roll in the same start request without a duration field.
- The pre-roll is processed by the existing live segmentation/inference pipeline before live audio, once and in order.
- Live PCM arriving during pre-roll processing is buffered within a defined bound and handled explicitly if the bound is exceeded.
- API validation, time semantics, and the request format are documented and covered by tests.


## Implementation status (reviewed 2026-10-28)

**Status: implemented; automated verification is present.** The earlier sentence above saying this is a plan only is stale and superseded by this status.

Implemented in `auditor_stt/serve/live/api.py`: legacy JSON start requests remain supported; multipart requests accept JSON `metadata` and optional binary `preroll`; `has_preroll` validation, a 10 MiB payload ceiling, non-empty/even-byte checks, and clear 4xx responses are enforced before session creation. The live session receives the PCM pre-roll, and `tests/test_live.py` covers multipart pre-roll processing before live audio. `docs/live-sessions.md` documents the PCM format and bounded live buffering.

**Remaining acceptance work:** run the current full Python test suite and OpenAPI/schema checks on the latest default branch; verify timeline/wall-clock boundary and overflow/cancellation/reconnect cases against the full test suite. End-to-end validation with the actual MediaMTX source and target deployment remains operational validation, not a code-completion claim.

**Contract:** pre-roll is mono 16 kHz signed 16-bit little-endian PCM; its duration is inferred from payload size. PiVoice sends it directly to Auditor. PiGateway coordinates session lifecycle only and must never proxy audio bytes.
