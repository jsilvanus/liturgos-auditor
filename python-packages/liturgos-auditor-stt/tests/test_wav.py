import os
import struct
import wave

import numpy as np
import pytest

from auditor_stt.serve.jobs.wav import open_pcm16_mono, write_pcm16_wav

RATE = 16000


def _chunk(chunk_id, body):
    """A RIFF chunk, padded to an even size like the spec requires."""
    padded = body + (b"\x00" if len(body) % 2 else b"")
    return chunk_id + struct.pack("<I", len(body)) + padded


def _fmt(rate=RATE, tag=1, channels=1, bits=16):
    block = channels * bits // 8
    return _chunk(b"fmt ", struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits))


def _wav_bytes(samples, rate=RATE, before=b"", after=b"", data_size=None, fmt=None):
    data = np.asarray(samples, dtype="<i2").tobytes()
    size = len(data) if data_size is None else data_size
    body = b"WAVE" + (fmt or _fmt(rate)) + before + b"data" + struct.pack("<I", size) + data + after
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _write(path, payload):
    path.write_bytes(payload)
    return path


SAMPLES = np.arange(-50, 50, dtype=np.int16) * 300


def test_roundtrip_through_write_pcm16_wav(tmp_path):
    t = np.arange(RATE) / RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    write_pcm16_wav(path, tone, RATE)

    with open_pcm16_mono(path) as wav:
        assert wav.sample_rate == RATE
        assert wav.num_samples == RATE
        assert wav.duration == pytest.approx(1.0)
        audio = wav.read(0, wav.num_samples)

    assert audio.dtype == np.float32
    assert type(audio) is np.ndarray  # a copy, not a view of the map
    np.testing.assert_allclose(audio, tone, atol=1 / 32768)


def test_write_pcm16_wav_clips_and_accepts_int16(tmp_path):
    write_pcm16_wav(tmp_path / "f.wav", np.array([2.0, -2.0, 0.0]), RATE)
    write_pcm16_wav(tmp_path / "i.wav", np.array([32767, -32768, 5], dtype=np.int16), RATE)
    with open_pcm16_mono(tmp_path / "f.wav") as wav:
        assert wav.read(0, 3).tolist() == [32767 / 32768, -1.0, 0.0]
    with open_pcm16_mono(tmp_path / "i.wav") as wav:
        assert wav.read(0, 3).tolist() == [32767 / 32768, -1.0, 5 / 32768]


def test_read_returns_the_requested_slice_scaled_to_unit_range(tmp_path):
    path = _write(tmp_path / "a.wav", _wav_bytes(SAMPLES))
    with open_pcm16_mono(path) as wav:
        assert wav.num_samples == 100
        np.testing.assert_array_equal(wav.read(10, 20), SAMPLES[10:20] / 32768.0)


def test_read_clamps_out_of_range_bounds(tmp_path):
    path = _write(tmp_path / "a.wav", _wav_bytes(SAMPLES))
    with open_pcm16_mono(path) as wav:
        assert len(wav.read(-5, 10)) == 10
        assert len(wav.read(98, 500)) == 2
        assert len(wav.read(5, 5)) == 0
        assert len(wav.read(50, 10)) == 0
        assert len(wav.read(200, 300)) == 0


def test_skips_list_chunk_before_data(tmp_path):
    info = _chunk(b"LIST", b"INFO" + _chunk(b"ISFT", b"Lavf60.16.100\x00"))
    path = _write(tmp_path / "ffmpeg.wav", _wav_bytes(SAMPLES, before=info))
    assert path.read_bytes().find(b"data") != 44  # the header is not the canonical 44 bytes

    with open_pcm16_mono(path) as wav:
        assert wav.num_samples == 100
        np.testing.assert_array_equal(wav.read(0, 100), SAMPLES / 32768.0)


def test_handles_odd_sized_chunk_padding(tmp_path):
    odd = _chunk(b"junk", b"abc")  # 3 bytes + 1 pad byte
    path = _write(tmp_path / "odd.wav", _wav_bytes(SAMPLES, before=odd))
    with open_pcm16_mono(path) as wav:
        assert wav.num_samples == 100
        np.testing.assert_array_equal(wav.read(0, 100), SAMPLES / 32768.0)


@pytest.mark.parametrize("placeholder", [0xFFFFFFFF, 0])
def test_placeholder_data_size_is_clamped_to_the_file(tmp_path, placeholder):
    # ffmpeg writing to a pipe cannot seek back to fix the size fields.
    info = _chunk(b"LIST", b"INFO" + _chunk(b"ISFT", b"Lavf\x00"))
    path = _write(tmp_path / "pipe.wav", _wav_bytes(SAMPLES, before=info, data_size=placeholder))
    with open_pcm16_mono(path) as wav:
        assert wav.num_samples == 100
        np.testing.assert_array_equal(wav.read(0, 100), SAMPLES / 32768.0)


def test_data_size_beyond_end_of_file_is_clamped(tmp_path):
    path = _write(tmp_path / "cut.wav", _wav_bytes(SAMPLES, data_size=10_000))
    with open_pcm16_mono(path) as wav:
        assert wav.num_samples == 100


def test_partial_trailing_byte_is_ignored(tmp_path):
    path = tmp_path / "odd_end.wav"
    path.write_bytes(_wav_bytes(SAMPLES, data_size=0xFFFFFFFF) + b"\x01")
    with open_pcm16_mono(path) as wav:
        assert wav.num_samples == 100


def test_declared_data_size_excludes_trailing_chunks(tmp_path):
    trailer = _chunk(b"LIST", b"INFOxxxx")
    path = _write(tmp_path / "trailer.wav", _wav_bytes(SAMPLES, after=trailer))
    with open_pcm16_mono(path) as wav:
        assert wav.num_samples == 100


def test_empty_data_chunk(tmp_path):
    path = _write(tmp_path / "empty.wav", _wav_bytes([]))
    wav = open_pcm16_mono(path)
    assert wav.num_samples == 0
    assert wav.duration == 0.0
    assert len(wav.read(0, 10)) == 0
    wav.close()


def test_uses_a_memory_map_not_a_copy(tmp_path):
    path = _write(tmp_path / "a.wav", _wav_bytes(SAMPLES))
    with open_pcm16_mono(path) as wav:
        assert isinstance(wav._data, np.memmap)


def test_close_releases_the_file_so_it_can_be_deleted(tmp_path):
    path = _write(tmp_path / "a.wav", _wav_bytes(SAMPLES))
    wav = open_pcm16_mono(path)
    wav.read(0, 10)
    wav.close()
    os.remove(path)  # PermissionError on Windows if the map were still held
    assert not path.exists()
    wav.close()  # idempotent


def test_read_after_close_raises(tmp_path):
    path = _write(tmp_path / "a.wav", _wav_bytes(SAMPLES))
    wav = open_pcm16_mono(path)
    wav.close()
    with pytest.raises(ValueError, match="closed"):
        wav.read(0, 10)


def test_rejects_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(b"\x00\x00" * 20)
    with pytest.raises(ValueError, match="16-bit PCM mono"):
        open_pcm16_mono(path)


def test_rejects_8_bit_and_float_and_other_bit_depths(tmp_path):
    with pytest.raises(ValueError, match="16-bit PCM mono"):
        open_pcm16_mono(_write(tmp_path / "u8.wav", _wav_bytes([], fmt=_fmt(bits=8))))
    with pytest.raises(ValueError, match="16-bit PCM mono"):
        open_pcm16_mono(_write(tmp_path / "f32.wav", _wav_bytes([], fmt=_fmt(tag=3, bits=32))))
    with pytest.raises(ValueError, match="16-bit PCM mono"):
        open_pcm16_mono(_write(tmp_path / "s24.wav", _wav_bytes([], fmt=_fmt(bits=24))))


def test_rejects_non_wav_and_structurally_broken_files(tmp_path):
    with pytest.raises(ValueError, match="RIFF"):
        open_pcm16_mono(_write(tmp_path / "text.wav", b"definitely not audio"))
    with pytest.raises(ValueError, match="RIFF"):
        open_pcm16_mono(_write(tmp_path / "tiny.wav", b"RIFF"))

    no_data = b"WAVE" + _fmt()
    with pytest.raises(ValueError, match="no data chunk"):
        open_pcm16_mono(_write(tmp_path / "nodata.wav", b"RIFF" + struct.pack("<I", len(no_data)) + no_data))

    data_first = b"WAVE" + _chunk(b"data", b"\x00\x00") + _fmt()
    with pytest.raises(ValueError, match="before the fmt chunk"):
        open_pcm16_mono(_write(tmp_path / "order.wav", b"RIFF" + struct.pack("<I", len(data_first)) + data_first))
