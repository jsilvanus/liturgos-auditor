# Word-level transcription

The inference API returns segment timestamps plus word timestamps.

Each segment has:

- `start`, `end`: segment timing in seconds
- `text`: segment text
- `words[]`: word timing records

Each word record has:

- `start`
- `end`
- `text`
- `probability`

The service requests `word_timestamps=True` from faster-whisper. This makes
word timing part of the service's canonical HTTP response rather than a
consumer-specific post-processing step.

Example:

```json
{
  "text": "moi maailma",
  "language": "fi",
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": "moi maailma",
      "words": [
        {"start": 0.0, "end": 0.4, "text": "moi", "probability": 0.99},
        {"start": 0.5, "end": 1.5, "text": " maailma", "probability": 0.98}
      ]
    }
  ]
}
```

This is suitable as the timing source for caption generation, transcript
navigation, and later semantic section analysis.
