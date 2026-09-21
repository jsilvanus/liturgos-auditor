# Video to WebVTT

`scripts/video-to-vtt.py` is a local testing/helper script that turns a video into a WebVTT subtitle file using a running `liturgos-auditor-stt` service.

The script processes the video incrementally: it extracts a bounded audio chunk, sends that chunk to STT, and appends the resulting VTT cues immediately. It does not extract the entire video's audio before transcription.

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on `PATH`
- a running STT service exposing `/inference` with word timestamps

## Usage

The simplest invocation writes a VTT file next to the video:

```bash
python3 scripts/video-to-vtt.py sermon.mp4
```

Or choose the output path:

```bash
python3 scripts/video-to-vtt.py sermon.mp4 sermon.vtt
```

The service defaults to `http://localhost:8090`. Override it with either:

```bash
AUDITOR_STT_URL=http://stt.example:8090 \
  python3 scripts/video-to-vtt.py sermon.mp4 sermon.vtt
```

or:

```bash
python3 scripts/video-to-vtt.py \
  --url http://stt.example:8090 \
  sermon.mp4 sermon.vtt
```

Finnish is the default language; it can be changed with `--language`.

## Chunking and incremental output

By default the script uses:

- 60 second nominal STT chunks;
- 5 seconds of overlap between consecutive chunks;
- VTT output appended after every successful chunk.

For example:

```
chunk 1:  0s -> 60s
chunk 2: 55s -> 115s
chunk 3: 110s -> 170s
...
```

The overlap is transcription context, not duplicated subtitle output. Each word is assigned to one output interval, so a word is written to the VTT only once.

The result is a normal WebVTT file that can be opened while transcription is still running. If a later chunk fails, the already completed part remains on disk.

Chunking can be adjusted:

```bash
python3 scripts/video-to-vtt.py \
  --chunk-seconds 90 \
  --overlap-seconds 5 \
  sermon.mp4 sermon.vtt
```

The STT API itself remains unchanged: each request is still an ordinary `POST /inference` containing one audio chunk. This keeps the service generic while the client owns batching, timeline offsets, overlap handling, and VTT presentation.

## Caption grouping

The script uses word-level timestamps from the STT response and groups them into readable cues. By default:

- cues are limited to about 7 seconds;
- sentence punctuation is preferred as a cue boundary;
- cues target two lines of about 42 characters each;
- cue timestamps are taken from the first and last word.

The grouping is intentionally a client-side concern. The STT service returns transcription timing; it does not make assumptions about subtitle presentation.

This is a helper for local use. Consumer applications such as SaarnaVideo can later use the same approach while adding their own batching, persistence, metadata, or publishing workflow.
