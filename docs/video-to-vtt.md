# video-to-vtt script

`scripts/video-to-vtt.py` transcribes a video with the service's [batch jobs API](batch-jobs.md) and writes the result as a subtitle or transcript file: VTT, SRT, YouTube Live Captions records, or the JSON transcript (`--format vtt|srt|youtube|json`). It uses only the Python standard library.

What it does, in order:

1. Pre-flight check against the service (skipped with `--resume`): a wrong API key or a service without jobs fails here, before any extraction or upload.
2. Extracts the audio with ffmpeg into a temporary 16 kHz mono 16-bit WAV (skipped with `--no-extract` and `--source-path`) and uploads it as a stream, so the file is never held in memory. With `--source-path` nothing is uploaded.
3. Prints the job id and polls the job until it ends, showing progress.
4. Fetches the result, writes the output file, and deletes the job on the server (unless `--keep-job`).

## Installation

Requirements:
- Python 3.9+
- `ffmpeg` on PATH (unless `--no-extract` or `--source-path`)
- A running `liturgos-auditor-stt` service with jobs enabled (`AUDITOR_STT_DATA_DIR` set on the service)

## Usage

### Basic usage

Transcribe a video and write VTT:

```bash
python3 scripts/video-to-vtt.py sermon.mp4
```

The output is `sermon.vtt`, written to the **current directory** (not next to the video). The default name is the video's name without extension plus `.vtt`, `.srt`, `.txt` (for `youtube`) or `.json`, depending on `--format`.

Specify the output path:

```bash
python3 scripts/video-to-vtt.py sermon.mp4 sermon.vtt
```

### What you see

Messages go to stderr. For an upload they look like:

```
Extracting audio: sermon.mp4
Submitting job to http://localhost:8090
Job 3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d submitted (resume with: --resume 3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d)
```

followed by progress lines (percentage, `(done/total chunks)` once the chunk plan exists, and `ETA <n>s` once the service can estimate it; on a terminal the line is updated in place), then `Fetching result` and `Wrote sermon.vtt`. The job id is printed as soon as the service has accepted the job.

### Service URL and authentication

Default service URL: `http://localhost:8090`, or the `AUDITOR_STT_URL` environment variable, or `--url`.

The API key is read from an environment variable, `AUDITOR_STT_API_KEY` by default; `--api-key-env` names a different variable (the option takes the variable's name, not the key). If the variable is not set the script sends no `Authorization` header, which works against a service that has no API key.

```bash
export AUDITOR_STT_API_KEY=your_key
python3 scripts/video-to-vtt.py sermon.mp4 sermon.vtt
```

```bash
AUDITOR_STT_URL=http://stt.example:8090 \
  AUDITOR_STT_API_KEY=your_key \
  python3 scripts/video-to-vtt.py sermon.mp4 sermon.vtt
```

```bash
export MY_STT_KEY=secret
python3 scripts/video-to-vtt.py \
  --url http://stt.example:8090 \
  --api-key-env MY_STT_KEY \
  sermon.mp4 sermon.vtt
```

## Command-line options

| Option | Type | Default | Notes |
|--------|------|---------|-------|
| `video` | Path | — | Video (or audio) file. Required unless `--source-path` or `--resume` is used; see the note on positional arguments below |
| `output` | Path | see [Basic usage](#basic-usage) | Output file. Required with `--resume` |
| `--language` | string | `fi` | STT language passed to the service (this script always sends a language; it does not use the service's default) |
| `--url` | URL | `AUDITOR_STT_URL`, else `http://localhost:8090` | Service base URL |
| `--api-key-env` | string | `AUDITOR_STT_API_KEY` | Name of the environment variable holding the API key |
| `--chunk-seconds` | number | `60` | Chunk size in seconds, passed to the service as is. The service accepts 5 to 300; anything else is refused with 422 and the script exits with code 1 |
| `--format` | string | `vtt` | `vtt`, `srt`, `youtube` or `json` |
| `--start-time` | ISO-8601 | — | Wall-clock time of media t=0 (UTC unless it carries an offset). Required for `--format youtube` (usage error otherwise) |
| `--max-cue-duration` | float | `7` | Maximum cue duration in seconds (VTT, SRT, YouTube) |
| `--max-line-chars` | int | `42` | Target maximum characters per subtitle line (VTT, SRT, YouTube) |
| `--poll-interval` | float | `3` | Polling interval in seconds |
| `--source-path` | string | — | Path of the file **on the server**, under the service's `AUDITOR_STT_MEDIA_ROOT`; nothing is uploaded and the script does not check the path itself. A relative path is resolved against the media root by the service |
| `--no-extract` | flag | — | Upload the file as is instead of extracting audio first |
| `--resume` | Job ID | — | Skip the pre-flight check and the submission; poll this existing job and fetch its result |
| `--keep-job` | flag | — | After a successful run, do not delete the job on the server. Has no effect on failed or cancelled jobs (see below) |
| `--json-out` | Path | — | After a successful run also save the JSON transcript |
| `--cancel-on-abort` | flag | — | On Ctrl-C cancel the job on the server instead of leaving it running |

**Positional arguments.** `video` and `output` are both optional positionals and are filled in order. With `--resume` or `--source-path`, a single positional is therefore read as `video`, not as `output`: `--resume ID sermon.vtt` stops with `OUTPUT is required with --resume` (exit code 2), and `--source-path ... out.vtt` ignores `out.vtt` and derives the output name from the source path. To set the output name with these options, give both positionals; the first one is not used (it can be the original file name), as in the examples below.

## Examples

### Different output formats

VTT (default):

```bash
python3 scripts/video-to-vtt.py sermon.mp4 sermon.vtt
```

SRT:

```bash
python3 scripts/video-to-vtt.py sermon.mp4 sermon.srt --format srt
```

YouTube Live Captions (requires start time):

```bash
python3 scripts/video-to-vtt.py sermon.mp4 sermon.txt \
  --format youtube \
  --start-time 2024-09-21T12:34:56Z
```

JSON:

```bash
python3 scripts/video-to-vtt.py sermon.mp4 sermon.json --format json
```

### Using source_path (no upload)

`--source-path` is a path on the **server**, inside the directory the service has as `AUDITOR_STT_MEDIA_ROOT` (for example a shared volume mounted read-only into the service container). The service reads the file itself; the script uploads nothing and needs no ffmpeg. If the service has no media root, or the path is not an existing file inside it, the service answers 422 and the script prints the reason and exits with code 1.

```bash
python3 scripts/video-to-vtt.py --source-path /media/sermons/sermon.mp4
```

This writes `sermon.vtt` in the current directory (named after the source file). To choose the output name, pass a placeholder video argument first:

```bash
python3 scripts/video-to-vtt.py --source-path /media/sermons/sermon.mp4 sermon.mp4 sermon-fi.vtt
```

### Resume a job

The job id is printed to stderr as soon as the job is accepted:

```
Job 3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d submitted (resume with: --resume 3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d)
```

Resume later (the first positional is not used but must be present, see above):

```bash
python3 scripts/video-to-vtt.py --resume 3f2a1d8e9c4b5a6f7e8d9c0b1a2f3e4d sermon.mp4 sermon.vtt
```

Use the same `--format` (and `--start-time` for `youtube`) as in the original run if you want the same kind of output. Resuming is useful if the script was killed or gave up waiting: the job runs on the server regardless of the script. It only works while the job still exists: a successful run deletes the job unless `--keep-job` was given, and finished jobs are purged after the service's `AUDITOR_STT_JOB_TTL_HOURS`. An unknown job id ends with `error: Job status request failed (404): No such job` and exit code 1.

### Smaller chunks for faster progress

A smaller `chunk_seconds` gives more frequent progress updates and lets live requests to the same service in between chunks more often:

```bash
python3 scripts/video-to-vtt.py sermon.mp4 sermon.vtt --chunk-seconds 30
```

## Retries and failures

The script separates what is safe to retry from what is not:

| Step | Retried | Not retried |
|------|---------|-------------|
| Pre-flight check | Connection refused, and 502/503/504 answers; 3 attempts | A 401 (`The service rejected the API key ...`) and any other error fail at once. A service without jobs answers 503, so it is retried and then fails with `error: Request failed after 3 attempts: Service check failed (503): Jobs are not configured ...` |
| Submitting the job (upload or `source_path`) | Only a **refused connection** (nothing was sent yet); up to 10 attempts with exponential backoff | Everything else, so a large upload is never silently sent twice and cannot create a duplicate job |
| Polling the status, fetching the result | Connection errors (refused, reset, broken pipe, timeout, disconnect) and 502/503/504 answers; up to 10 attempts with exponential backoff starting at 0.5 s (about four minutes of waiting in total) | Other answers, for example 404 when the job no longer exists |

Every request has a 30 second socket timeout. While polling is retried the job keeps running on the server; if the attempts run out the script prints `error: Request failed after 10 attempts: ...` and exits with code 1, and you can pick the job up later with `--resume`.

If the service refuses an upload (wrong or missing key, file over `AUDITOR_STT_MAX_UPLOAD_MB`), it closes the connection; the script then prints the reason it could read, or `The service closed the connection during the upload (wrong or missing API key, or the file exceeds its upload limit?)`.

## Partial results and exit codes

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. The output file is written; the job is deleted on the server unless `--keep-job` (a failed deletion only prints a warning) |
| `1` | Error: printed as `error: ...`. For example the pre-flight or a submission was refused, the service could not be reached after the retries, a request failed, or ffmpeg failed |
| `2` | Usage error (invalid or missing command-line arguments, video file not found) |
| `3` | The job **failed or was cancelled** on the server. The script prints `Error: <message>` when the job has one and `Completed X/Y chunks; the job record stays on the server until it expires`, and tries to save the partial result (see below) |
| `130` | Interrupted with Ctrl-C |

### Failed and cancelled jobs

A failed or cancelled job is **always kept** on the server; `--keep-job` only matters after success. Its finished chunks stay available until the job expires (`AUDITOR_STT_JOB_TTL_HOURS` after it ended, 72 by default), so you can inspect it with `GET /v1/jobs/{id}` and fetch the finished part with `GET /v1/jobs/{id}/result?partial=1`. See the [batch jobs API](batch-jobs.md).

The script fetches that partial result itself, in the requested format, and writes it next to the output name:

```
sermon.vtt.partial
```

If fetching it fails, the script says so and still exits with code 3. To use the partial result under the real name:

```bash
mv sermon.vtt.partial sermon.vtt
```

### Ctrl-C

Without `--cancel-on-abort`, Ctrl-C leaves the job running on the server, prints `Interrupted. The job keeps running on the server; resume with: --resume <id>`, and exits with 130. With `--cancel-on-abort` it sends a `DELETE` for the job (a running job is stopped between chunks and then removed), and exits with 130. Both apply once the job id is known, that is after the job was accepted.

## Cue grouping

VTT, SRT and YouTube cue grouping happens in the service; the script only passes `--max-cue-duration` and `--max-line-chars` as query parameters and writes what it gets back. They have no effect on `--format json`.
