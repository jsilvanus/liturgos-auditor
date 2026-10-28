"""Live sessions (pull type): clock, segmenter, session loop, SSE routes and the fleet stream."""

import json
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from auditor_stt.serve.app import create_app
from auditor_stt.serve.jobs.fleet import FleetConfig, FleetStripper, FleetUnavailableError
from auditor_stt.serve.live.clock import WallClock, iso
from auditor_stt.serve.live.segmenter import Segmenter, energy_speech_ranges
from auditor_stt.serve.live.session import LiveConfig, LiveError, LiveManager, check_source
from auditor_stt.serve.live.source import FleetPcmStream, stream_spec
from auditor_stt.serve.queue import InferenceQueue

RATE = 16000
SOURCE = "rtsp://mediamtx.test:8554/live/main?user=a&pass=sekret"


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("AUDITOR_STT_"):
            monkeypatch.delenv(name, raising=False)


def tone(seconds, amplitude=0.3, freq=220.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def quiet(seconds):
    return np.zeros(int(seconds * RATE), dtype=np.float32)


def pcm(samples):
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


# --- clock -----------------------------------------------------------------


def test_clock_maps_samples_to_wall_time_from_the_anchor():
    now = [1000.0]
    clock = WallClock(RATE, now=lambda: now[0])
    assert clock.check(8000, caught_up=True) is None  # first data anchors
    assert clock.time_of(8000) == 1000.0
    assert clock.time_of(8000 + RATE) == 1001.0


def test_clock_resets_on_drift_only_when_caught_up():
    now = [1000.0]
    clock = WallClock(RATE, now=lambda: now[0])
    clock.check(0, caught_up=True)
    now[0] = 1005.0  # 5 s of wall time but only 1 s of samples: the source skipped
    assert clock.check(RATE, caught_up=False) is None  # a backlog proves nothing
    assert clock.time_of(RATE) == 1001.0
    drift = clock.check(RATE, caught_up=True)
    assert drift == pytest.approx(4.0)
    assert clock.time_of(RATE) == 1005.0
    now[0] = 1005.5  # small jitter is kept
    assert clock.check(RATE + 8000, caught_up=True) is None


def test_iso_has_milliseconds_and_utc():
    assert iso(1759656600.25) == "2025-10-05T09:30:00.250Z"
    assert iso(1759656600.9996) == "2025-10-05T09:30:01.000Z"


# --- segmenter ---------------------------------------------------------------


def feed_all(segmenter, samples, block=4000):
    out = []
    for i in range(0, len(samples), block):
        out.extend(segmenter.feed(samples[i : i + block]))
    return out


def test_energy_ranges_find_bursts():
    audio = np.concatenate([quiet(1), tone(1), quiet(1), tone(0.5), quiet(1)])
    ranges = energy_speech_ranges(audio)
    assert len(ranges) == 2
    assert ranges[0][0] == pytest.approx(1 * RATE, abs=0.1 * RATE)
    assert ranges[1][1] == pytest.approx(3.5 * RATE, abs=0.3 * RATE)


def test_segments_end_at_pauses_and_silence_is_dropped():
    audio = np.concatenate([quiet(2), tone(2), quiet(1), tone(2), quiet(3)])
    segmenter = Segmenter(RATE, speech_ranges=energy_speech_ranges)
    segments = feed_all(segmenter, audio)
    assert len(segments) == 2
    first, second = segments
    assert first.start_sample == pytest.approx(2 * RATE, abs=0.3 * RATE)
    assert (first.end_sample - first.start_sample) / RATE == pytest.approx(2.3, abs=0.4)
    assert second.start_sample > first.end_sample
    assert len(first.samples) == first.end_sample - first.start_sample
    assert segmenter.flush() is None


def test_long_speech_is_cut_inside_a_short_gap_not_through_speech():
    audio = np.concatenate([tone(6), quiet(0.35), tone(6), quiet(3)])
    segmenter = Segmenter(RATE, max_seconds=10, speech_ranges=energy_speech_ranges)
    segments = feed_all(segmenter, audio)
    assert len(segments) == 2
    gap_mid = (6 + 0.175) * RATE
    assert segments[0].end_sample == pytest.approx(gap_mid, abs=0.2 * RATE)
    assert segments[1].start_sample >= segments[0].end_sample


def test_unbroken_speech_is_cut_at_the_maximum():
    segmenter = Segmenter(RATE, max_seconds=5, speech_ranges=energy_speech_ranges)
    segments = feed_all(segmenter, tone(12))
    assert [(s.end_sample - s.start_sample) / RATE for s in segments] == [5.0, 5.0]
    assert segmenter.flush() is not None  # the last 2 s


def test_flush_returns_the_open_segment():
    segmenter = Segmenter(RATE, speech_ranges=energy_speech_ranges)
    assert feed_all(segmenter, np.concatenate([quiet(1), tone(1.5)])) == []
    tail = segmenter.flush()
    assert tail is not None and (tail.end_sample - tail.start_sample) / RATE >= 1.4


def test_a_short_word_is_emitted_after_a_long_pause():
    segmenter = Segmenter(RATE, speech_ranges=energy_speech_ranges)
    segments = feed_all(segmenter, np.concatenate([tone(0.4), quiet(3)]))
    assert len(segments) == 1


# --- session -------------------------------------------------------------------


class Host:
    model_id = "stub"
    loaded = True

    def load(self):
        pass

    def __init__(self):
        self.calls = []

    def transcribe_array(self, samples, language, **options):
        self.calls.append({"seconds": len(samples) / RATE, "language": language, **options})
        return {"text": f" words {len(self.calls)} ", "language": language, "segments": []}


class ListStream:
    def __init__(self, data, hold=None):
        self._data = data
        self._pos = 0
        self.closed = False
        self._hold = hold

    def read(self, size):
        if self.closed:
            return b""
        if self._pos >= len(self._data):
            if self._hold is not None:
                self._hold.wait(5)
            return b""
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self):
        self.closed = True


def speech_audio():
    return np.concatenate([quiet(1), tone(2), quiet(1), tone(2), quiet(1.5)])


def make_client(host, streams, config=None, now=time.time):
    config = config or LiveConfig(source_hosts=["mediamtx.test"], reconnect_seconds=0.2)
    attempts = []

    def factory(source, session_id, attempt):
        attempts.append((source, session_id, attempt))
        item = streams.pop(0) if streams else None
        if item is None:
            raise OSError("no stream")
        return item

    state = {}

    def build(app):
        manager = LiveManager(
            config, lambda: app.state.model_host, app.state.queue, stream_factory=factory,
            speech_ranges=energy_speech_ranges, now=now,
        )
        state["manager"] = manager
        return manager

    app = create_app(model_host=host, queue=InferenceQueue(max_queue=8), api_key="")
    app.state.live = build(app)
    return TestClient(app), state["manager"], attempts


def read_events(client, session_id, until="ended"):
    events = []
    with client.stream("GET", f"/v1/live/{session_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        current = {}
        for line in response.iter_lines():
            if line.startswith(":"):
                continue
            if line == "":
                if current:
                    events.append(current)
                    if current.get("event") == "status" and json.loads(current["data"]).get("state") == until:
                        break
                current = {}
                continue
            key, _, value = line.partition(": ")
            current[key] = value
    return events


def wait_ended(client, session_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get(f"/v1/live/{session_id}").json()
        if info["status"] == "ended":
            return info
        time.sleep(0.05)
    raise AssertionError("session did not end")


def test_session_publishes_transcripts_with_absolute_times():
    host = Host()
    start = 1759656000.0
    # A frozen wall clock against a stream that is read instantly: keep the anchor from moving.
    config = LiveConfig(source_hosts=["mediamtx.test"], reconnect_seconds=0.2, clock_drift_seconds=1e9)
    client, manager, attempts = make_client(host, [ListStream(pcm(speech_audio()))], config, now=lambda: start)
    with client:
        response = client.post("/v1/live", json={"source": SOURCE, "language": "fi", "client_ref": "svc-1"})
        assert response.status_code == 202
        body = response.json()
        assert body["events_url"] == f"/v1/live/{body['id']}/events"
        events = read_events(client, body["id"])
        info = wait_ended(client, body["id"])
    transcripts = [json.loads(e["data"]) for e in events if e["event"] == "transcript"]
    assert [t["sequence"] for t in transcripts] == [1, 2]
    assert [t["text"] for t in transcripts] == ["words 1", "words 2"]
    # The first sample arrived at `start` (frozen clock): speech begins ~1 s in.
    assert transcripts[0]["wall_start"].endswith("Z")
    assert 1.7 < transcripts[0]["end"] - transcripts[0]["start"] < 2.8
    # The anchor is the arrival of the first 0.25 s block, speech starts at 1 s (less a 0.15 s lead-in).
    assert transcripts[0]["start"] == pytest.approx(0.6, abs=0.05)
    assert transcripts[0]["wall_end"] < transcripts[1]["wall_start"]
    assert transcripts[0]["wall_start"] >= iso(start + 0.5)
    assert transcripts[1]["wall_start"] >= iso(start + 3.5)
    assert info["client_ref"] == "svc-1"
    assert info["transcripts"] == 2
    ids = [int(e["id"]) for e in events]
    assert ids == sorted(ids)
    # Language and the carry-over prompt reach the model.
    assert host.calls[0]["language"] == "fi" and host.calls[0]["prompt"] is None
    assert "words 1" in host.calls[1]["prompt"]


def test_events_can_be_resumed_with_last_event_id():
    host = Host()
    client, manager, _ = make_client(host, [ListStream(pcm(speech_audio()))])
    with client:
        sid = client.post("/v1/live", json={"source": SOURCE}).json()["id"]
        wait_ended(client, sid)
        everything = read_events(client, sid)
        first_id = everything[0]["id"]
        with client.stream("GET", f"/v1/live/{sid}/events", headers={"Last-Event-ID": first_id}) as response:
            resumed = [line for line in response.iter_lines() if line.startswith("id: ")]
    assert len(resumed) == len(everything) - 1


def test_stop_closes_the_stream_and_ends_the_session():
    host = Host()
    hold = threading.Event()
    stream = ListStream(pcm(tone(1)), hold=hold)
    client, manager, _ = make_client(host, [stream])
    with client:
        sid = client.post("/v1/live", json={"source": SOURCE}).json()["id"]
        time.sleep(0.2)
        response = client.delete(f"/v1/live/{sid}")
        assert response.status_code == 200
        hold.set()
        info = wait_ended(client, sid)
    assert info["reason"] == "stopped"
    assert stream.closed


def test_reconnects_after_the_stream_ends_and_keeps_the_sequence():
    host = Host()
    first = ListStream(pcm(np.concatenate([quiet(0.5), tone(2), quiet(1)])))
    second = ListStream(pcm(np.concatenate([quiet(0.5), tone(2), quiet(1)])))
    config = LiveConfig(source_hosts=["mediamtx.test"], reconnect_seconds=30)
    client, manager, attempts = make_client(host, [first, second], config)
    with client:
        sid = client.post("/v1/live", json={"source": SOURCE}).json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and manager.get(sid).transcripts < 2:
            time.sleep(0.05)
        events = list(manager.get(sid).events)
        client.delete(f"/v1/live/{sid}")
    assert [a[2] for a in attempts[:2]] == [1, 2]
    assert any(e[1] == "status" and e[2]["state"] == "reconnecting" for e in events)
    assert [e[2]["sequence"] for e in events if e[1] == "transcript"] == [1, 2]


def test_gives_up_when_the_stream_cannot_be_reopened():
    host = Host()
    client, manager, attempts = make_client(host, [])
    with client:
        sid = client.post("/v1/live", json={"source": SOURCE}).json()["id"]
        info = wait_ended(client, sid)
    assert info["reason"] == "source_lost"
    errors = [e[2]["message"] for e in manager.get(sid).events if e[1] == "error"]
    assert errors and all("sekret" not in m for m in errors)
    assert len(attempts) >= 2


def test_old_segments_are_dropped_not_transcribed():
    class Slow(Host):
        def transcribe_array(self, samples, language, **options):
            time.sleep(0.4)
            return super().transcribe_array(samples, language, **options)

    host = Slow()
    config = LiveConfig(source_hosts=["mediamtx.test"], reconnect_seconds=0.2, max_lag_seconds=0.1)
    audio = np.concatenate([quiet(0.5), tone(1.5), quiet(1), tone(1.5), quiet(1), tone(1.5), quiet(2)])
    client, manager, _ = make_client(host, [ListStream(pcm(audio))], config)
    with client:
        sid = client.post("/v1/live", json={"source": SOURCE}).json()["id"]
        wait_ended(client, sid)
    states = [e[2].get("state") for e in manager.get(sid).events if e[1] == "status"]
    assert "dropped" in states
    assert len(host.calls) < 3


def test_clock_reset_status_when_the_source_skips():
    host = Host()
    wall = [1000.0]
    data = pcm(np.concatenate([quiet(0.5), tone(1.5), quiet(1)]))

    def now():
        wall[0] += 0.5  # each look at the clock costs half a second: more than the stream delivers
        return wall[0]

    client, manager, _ = make_client(host, [ListStream(data)], now=now)
    with client:
        sid = client.post("/v1/live", json={"source": SOURCE}).json()["id"]
        wait_ended(client, sid)
    assert any(e[2].get("state") == "clock_reset" for e in manager.get(sid).events if e[1] == "status")


# --- routes and validation --------------------------------------------------------


def test_source_validation():
    hosts = ["mediamtx.*"]
    check_source("rtsp://mediamtx.lan/a", hosts)
    check_source("srt://mediamtx.lan:8890?streamid=read:a", hosts)
    for url, status in (
        ("http://mediamtx.lan/a", 422),
        ("file:///etc/passwd", 422),
        ("rtsp://evil.example/a", 403),
        ("", 422),
    ):
        with pytest.raises(LiveError) as caught:
            check_source(url, hosts)
        assert caught.value.status == status
    with pytest.raises(LiveError) as caught:
        check_source("rtsp://mediamtx.lan/a", [])
    assert caught.value.status == 403


def test_multipart_preroll_is_processed_before_live_audio():
    class CapturingHost(Host):
        def transcribe_array(self, samples, language, **options):
            self.calls.append({"seconds": len(samples) / RATE, "rms": float(np.sqrt(np.mean(samples ** 2))), "language": language, **options})
            return {"text": f" words {len(self.calls)} ", "language": language, "segments": []}

    host = CapturingHost()
    config = LiveConfig(source_hosts=["mediamtx.test"], reconnect_seconds=0.2, clock_drift_seconds=1e9)
    preroll = pcm(np.concatenate([quiet(0.5), tone(2), quiet(1.5)]))
    live = pcm(np.concatenate([quiet(0.5), tone(2), quiet(2)]))
    client, manager, _ = make_client(host, [ListStream(live)], config, now=lambda: 1759656000.0)
    metadata = {
        "source": SOURCE,
        "language": "fi",
        "client_ref": "pivoice-main",
        "has_preroll": True,
        "preroll_start_ms": 1759655990000,
    }
    with client:
        response = client.post(
            "/v1/live",
            data={"metadata": json.dumps(metadata)},
            files={"preroll": ("preroll.pcm", preroll, "application/octet-stream")},
        )
        assert response.status_code == 202, response.text
        info = wait_ended(client, response.json()["id"])
    assert info["transcripts"] >= 2
    # The first transcribed segment comes from pre-roll; live audio follows it.
    assert host.calls[0]["seconds"] == pytest.approx(2.3, abs=0.4)
    assert host.calls[1]["seconds"] == pytest.approx(2.3, abs=0.4)
    transcripts = [event[2] for event in manager.get(response.json()["id"]).events if event[1] == "transcript"]
    assert transcripts and all(event["start"] >= 0 for event in transcripts)
    # The wall clock follows the supplied capture-time start, not HTTP/session arrival.
    assert transcripts[0]["wall_start"] >= iso(1759655990.0)
    assert transcripts[0]["wall_start"] < iso(1759655993.0)


def test_preroll_request_validation_happens_before_session_creation():
    host = Host()
    client, manager, _ = make_client(host, [ListStream(b""), ListStream(b"")])
    with client:
        missing = client.post(
            "/v1/live",
            files={"metadata": (None, json.dumps({"source": SOURCE, "has_preroll": True}))},
        )
        assert missing.status_code == 422
        empty = client.post(
            "/v1/live",
            data={"metadata": json.dumps({"source": SOURCE, "has_preroll": True})},
            files={"preroll": ("preroll.pcm", b"", "application/octet-stream")},
        )
        assert empty.status_code == 422
        odd = client.post(
            "/v1/live",
            data={"metadata": json.dumps({"source": SOURCE, "has_preroll": True})},
            files={"preroll": ("preroll.pcm", b"a", "application/octet-stream")},
        )
        assert odd.status_code == 422
        unexpected = client.post(
            "/v1/live",
            data={"metadata": json.dumps({"source": SOURCE, "has_preroll": False})},
            files={"preroll": ("preroll.pcm", b"ab", "application/octet-stream")},
        )
        assert unexpected.status_code == 422
        malformed = client.post(
            "/v1/live",
            data={"metadata": "{not json"},
            files={"preroll": ("preroll.pcm", b"\\x00\\x00", "application/octet-stream")},
        )
        assert malformed.status_code == 422
        assert manager.active_sessions() == []


def test_json_live_start_remains_backward_compatible_and_rejects_preroll_flag():
    client, manager, _ = make_client(Host(), [ListStream(b""), ListStream(b"")])
    with client:
        legacy = client.post("/v1/live", json={"source": SOURCE})
        assert legacy.status_code == 202
        client.delete(f"/v1/live/{legacy.json()['id']}")
        flagged = client.post("/v1/live", json={"source": SOURCE, "has_preroll": True})
        assert flagged.status_code == 422


def test_routes_report_disabled_unknown_and_busy():
    client = TestClient(create_app(model_host=Host(), api_key=""))
    with client:
        assert client.post("/v1/live", json={"source": SOURCE}).status_code == 503
    host = Host()
    config = LiveConfig(source_hosts=["mediamtx.test"], max_sessions=1, reconnect_seconds=5)
    hold = threading.Event()
    client, manager, _ = make_client(host, [ListStream(b"", hold=hold), ListStream(b"")], config)
    with client:
        assert client.get("/v1/live/nope").status_code == 404
        assert client.post("/v1/live", json={"source": "http://x/y"}).status_code == 422
        assert client.post("/v1/live", json={"source": "rtsp://elsewhere/x"}).status_code == 403
        first = client.post("/v1/live", json={"source": SOURCE})
        assert first.status_code == 202
        assert client.post("/v1/live", json={"source": SOURCE}).status_code == 429
        assert [s["id"] for s in client.get("/v1/live").json()["sessions"]] == [first.json()["id"]]
        hold.set()
        client.delete(f"/v1/live/{first.json()['id']}")


def test_live_routes_need_the_api_key():
    client = TestClient(create_app(model_host=Host(), api_key="secret"))
    with client:
        assert client.post("/v1/live", json={"source": SOURCE}).status_code == 401
        assert client.get("/v1/live").status_code == 401


def test_config_from_env():
    config = LiveConfig.from_env(
        {
            "AUDITOR_STT_LIVE_SOURCE_HOSTS": "MediaMTX, 10.1.*",
            "AUDITOR_STT_LIVE_REQUIRES": "net:mediamtx",
            "AUDITOR_STT_MAX_LIVE_SESSIONS": "4",
            "AUDITOR_STT_LIVE_MAX_SEGMENT_SECONDS": "8",
        }
    )
    assert config.source_hosts == ["mediamtx", "10.1.*"]
    assert config.requires == ["net:mediamtx"]
    assert config.max_sessions == 4 and config.max_segment_seconds == 8.0
    assert LiveConfig.from_env({}).source_hosts == []


# --- the fleet stream ----------------------------------------------------------------


class FakeFleet:
    def __init__(self, pcm_bytes=b"", behaviour="ok"):
        self.pcm = pcm_bytes
        self.behaviour = behaviour
        self.requests = []
        self.specs = {}

    def handler(self, request):
        path = request.url.path
        self.requests.append((request.method, path))
        if self.behaviour == "down":
            return httpx.Response(503)
        if path == "/v1/jobs" and request.method == "POST":
            spec = json.loads(request.content)
            if self.behaviour == "reject":
                return httpx.Response(422, json={"error": {"message": f"bad {spec['inputs'][0]['uri']}"}})
            self.specs[spec["id"]] = spec
            return httpx.Response(201, json={"id": spec["id"], "state": "running"})
        if path.endswith("/stdout"):
            return httpx.Response(200, stream=httpx.ByteStream(self.pcm), headers={"content-type": "application/octet-stream"})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    def stripper(self):
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return FleetStripper(FleetConfig(mode="fleet", url="http://fleet.test", token="t", public_url="http://a.test"), client=client)


def test_stream_spec_is_a_stdout_stream_job():
    spec = stream_spec("auditor-live-x-1", SOURCE, ["net:mediamtx"])
    assert spec["kind"] == "stream" and spec["stdout"] is True
    assert spec["requires"] == ["net:mediamtx"]
    assert spec["inputs"] == [{"name": "src", "uri": SOURCE}]
    args = spec["ffmpeg"]["args"]
    assert args[:2] == ["-rtsp_transport", "tcp"]
    assert SOURCE not in args and "{{input:src}}" in args
    assert args[-3:] == ["s16le", "pipe:1"][-3:] or args[-2:] == ["s16le", "pipe:1"]
    assert "-rtsp_transport" not in stream_spec("x", "srt://h:8890?streamid=a")["ffmpeg"]["args"]


def test_fleet_stream_reads_stdout_and_cancels_on_close():
    fleet = FakeFleet(pcm(tone(1)))
    stream = FleetPcmStream(fleet.stripper(), "auditor-live-x-1", SOURCE, requires=["net:mediamtx"])
    data = b""
    while True:
        chunk = stream.read(8000)
        if not chunk:
            break
        data += chunk
    assert len(data) == RATE * 2
    stream.close()
    stream.close()  # idempotent
    assert ("DELETE", "/v1/jobs/auditor-live-x-1") in fleet.requests
    assert fleet.specs["auditor-live-x-1"]["requires"] == ["net:mediamtx"]


def test_fleet_stream_errors_are_scrubbed_and_unavailable_is_distinct():
    with pytest.raises(OSError) as caught:
        FleetPcmStream(FakeFleet(behaviour="reject").stripper(), "id-1", SOURCE)
    assert "sekret" not in str(caught.value) and "<url>" in str(caught.value)
    with pytest.raises(FleetUnavailableError):
        FleetPcmStream(FakeFleet(behaviour="down").stripper(), "id-2", SOURCE)


def test_manager_falls_back_to_local_ffmpeg_when_the_fleet_is_down(monkeypatch):
    opened = []

    class Local:
        def __init__(self, url):
            opened.append(url)

    monkeypatch.setattr("auditor_stt.serve.live.session.LocalPcmStream", Local)
    config = LiveConfig(source_hosts=["mediamtx.test"])
    manager = LiveManager(config, lambda: None, None, fleet=FakeFleet(behaviour="down").stripper())
    manager.open_stream(SOURCE, "s1", 1)
    assert opened == [SOURCE]
    strict = LiveManager(config, lambda: None, None, fleet=FakeFleet(behaviour="down").stripper(), fleet_fallback=False)
    with pytest.raises(FleetUnavailableError):
        strict.open_stream(SOURCE, "s1", 1)


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_local_stream_decodes_to_pcm16_mono_16k(tmp_path):
    import subprocess

    from auditor_stt.serve.live.source import LocalPcmStream

    path = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-ar", "44100", "-ac", "2", str(path)],
        check=True,
    )
    stream = LocalPcmStream(str(path))
    data = b""
    while True:
        chunk = stream.read(8000)
        if not chunk:
            break
        data += chunk
    stream.close()
    assert len(data) == pytest.approx(1.5 * RATE * 2, abs=RATE * 0.1)
    assert np.abs(np.frombuffer(data, dtype="<i2")).max() > 1000


# --- delay ---------------------------------------------------------------------------


def test_delay_holds_transcripts_until_wall_end_plus_delay():
    host = Host()
    clock = [1759656000.0]
    config = LiveConfig(source_hosts=["mediamtx.test"], reconnect_seconds=0.2, clock_drift_seconds=1e9)
    client, manager, _ = make_client(host, [ListStream(pcm(speech_audio()))], config, now=lambda: clock[0])
    with client:
        body = client.post("/v1/live", json={"source": SOURCE, "delay_seconds": 20}).json()
        assert body["delay_seconds"] == 20
        session = manager.get(body["id"])
        deadline = time.monotonic() + 5
        while len(host.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.4)  # the transcribed text is waiting, not published
        assert len(host.calls) == 2
        assert not [e for e in session.events if e[1] == "transcript"]
        assert client.get(f"/v1/live/{body['id']}").json()["status"] != "ended"
        clock[0] += 10  # first segment ends ~3 s in: released at ~23 s; second ~6.5 s: ~26.5 s
        time.sleep(0.5)
        assert not [e for e in session.events if e[1] == "transcript"]
        clock[0] += 30
        info = wait_ended(client, body["id"])
    sent = [e[2] for e in session.events if e[1] == "transcript"]
    assert [t["sequence"] for t in sent] == [1, 2]
    assert all(t["delay_seconds"] == 20 for t in sent)
    # Times stay those of the source, not of the delayed publish.
    assert sent[0]["wall_start"] >= iso(1759656000.0 + 0.5)
    assert info["transcripts"] == 2


def test_delay_defaults_and_limits():
    host = Host()
    config = LiveConfig(
        source_hosts=["mediamtx.test"], reconnect_seconds=0.2, default_delay_seconds=2.0, max_delay_seconds=30.0
    )
    client, manager, _ = make_client(host, [ListStream(b"")], config)
    with client:
        assert client.post("/v1/live", json={"source": SOURCE, "delay_seconds": 31}).status_code == 422
        assert client.post("/v1/live", json={"source": SOURCE, "delay_seconds": -1}).status_code == 422
        created = client.post("/v1/live", json={"source": SOURCE})
        assert created.status_code == 202
        assert created.json()["delay_seconds"] == 2.0
        client.delete(f"/v1/live/{created.json()['id']}")


def test_delay_config_from_env():
    config = LiveConfig.from_env(
        {"AUDITOR_STT_LIVE_DELAY_SECONDS": "5", "AUDITOR_STT_LIVE_MAX_DELAY_SECONDS": "60"}
    )
    assert config.default_delay_seconds == 5.0 and config.max_delay_seconds == 60.0
    assert LiveConfig.from_env({}).default_delay_seconds == 0.0
