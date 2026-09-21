"""Memory-mapped access to 16-bit mono PCM WAV files.

Batch jobs normalise their input to a WAV on disk and then read one chunk at
a time; a 12 hour file is over 1 GB, so the samples are mapped, never loaded.
The RIFF chunks are walked rather than assuming a 44-byte header: ffmpeg adds
a LIST/INFO chunk, and when it writes to a pipe it cannot seek back, so the
data-chunk size is left as a placeholder (0 or 0xFFFFFFFF) and the real
length is whatever remains of the file.
"""

import os
import struct
import wave

import numpy as np

_PLACEHOLDER_SIZES = (0, 0xFFFFFFFF)


def _parse_header(path):
    """Return (sample_rate, data_offset, num_samples) for a PCM16 mono WAV."""
    file_size = os.path.getsize(path)
    sample_rate = None
    with open(path, "rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError("Not a RIFF/WAVE file")

        while True:
            header = f.read(8)
            if len(header) < 8:
                raise ValueError("WAV file has no data chunk")
            chunk_id, size = struct.unpack("<4sI", header)
            body = f.tell()

            if chunk_id == b"fmt ":
                fields = f.read(16)
                if size < 16 or len(fields) < 16:
                    raise ValueError("WAV fmt chunk is too short")
                tag, channels, rate, _byte_rate, _block_align, bits = struct.unpack("<HHIIHH", fields)
                if tag != 1 or channels != 1 or bits != 16 or rate <= 0:
                    raise ValueError(
                        f"Expected 16-bit PCM mono WAV, got format tag {tag}, "
                        f"{channels} channel(s), {bits}-bit, {rate} Hz"
                    )
                sample_rate = rate
            elif chunk_id == b"data":
                if sample_rate is None:
                    raise ValueError("WAV data chunk comes before the fmt chunk")
                remaining = file_size - body
                if size in _PLACEHOLDER_SIZES or size > remaining:
                    size = remaining  # streamed or truncated file: trust the file size
                return sample_rate, body, size // 2

            f.seek(body + size + (size & 1))  # chunks are padded to an even size


class Pcm16Wav:
    def __init__(self, path):
        self.path = str(path)
        self.sample_rate, offset, self.num_samples = _parse_header(self.path)
        self.duration = self.num_samples / self.sample_rate
        if self.num_samples:
            self._data = np.memmap(self.path, dtype="<i2", mode="r", offset=offset, shape=(self.num_samples,))
        else:
            self._data = np.zeros(0, dtype="<i2")  # mmap cannot map zero bytes

    def read(self, start_sample, end_sample):
        """Samples [start, end) as float32 in [-1, 1]; out-of-range bounds are clamped."""
        if self._data is None:
            raise ValueError("WAV file is closed")
        start = max(0, int(start_sample))
        end = min(self.num_samples, int(end_sample))
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        # asarray with a dtype copies into a plain ndarray, so no view of the map escapes.
        samples = np.asarray(self._data[start:end], dtype=np.float32)
        samples /= 32768.0
        return samples

    def close(self):
        """Release the mapping; Windows cannot delete a file that is still mapped."""
        data, self._data = self._data, None
        mapping = getattr(data, "_mmap", None)
        del data
        if mapping is not None:
            try:
                mapping.close()
            except BufferError:
                pass  # a caller still holds a view; the mapping goes when that does

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def open_pcm16_mono(path):
    return Pcm16Wav(path)


def write_pcm16_wav(path, samples, sample_rate):
    """Write mono PCM16; float input is taken as [-1, 1], int16 input is written as is."""
    array = np.asarray(samples)
    if array.dtype != np.int16:
        array = np.clip(np.rint(array * 32768.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(array.astype("<i2").tobytes())
