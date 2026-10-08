import asyncio
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event

import pytest
from fastapi.testclient import TestClient

import app as app_module


FakeSynthesizer = Callable[[str, str], bytes]
FakeStreamer = Callable[[str, str, Event], Iterator[bytes]]


@contextmanager
def open_test_app(
    monkeypatch: pytest.MonkeyPatch,
    synthesizer: FakeSynthesizer,
    *,
    streamer: FakeStreamer | None = None,
    queue_size: int = 10,
) -> Iterator[TestClient]:
    monkeypatch.setattr(app_module, "generate_wav", synthesizer)
    monkeypatch.setattr(
        app_module,
        "stream_audio_chunks",
        streamer
        or (lambda text, voice, _stop: iter([synthesizer(text, voice)])),
    )
    monkeypatch.setattr(app_module, "get_sample_rate", lambda: 24_000)
    monkeypatch.setattr(app_module, "get_tts", lambda: None)
    monkeypatch.setattr(app_module, "MAX_QUEUE_SIZE", queue_size)
    with TestClient(app_module.app) as client:
        yield client


def receive_json_until(websocket, predicate: Callable[[dict], bool]) -> dict:
    while True:
        message = websocket.receive_json()
        if predicate(message):
            return message


def collect_streams(
    websocket, count: int
) -> tuple[list[dict], list[tuple[dict, bytes]], list[dict]]:
    starts = []
    pending_chunks = []
    chunks = []
    completions = []
    while len(completions) < count:
        message = websocket.receive()
        if message.get("text") is not None:
            payload = json.loads(message["text"])
            if payload["type"] == "audio_start":
                starts.append(payload)
            elif payload["type"] == "audio_chunk":
                pending_chunks.append(payload)
            elif payload["type"] == "audio_complete":
                completions.append(payload)
        elif message.get("bytes") is not None:
            chunks.append((pending_chunks.pop(0), message["bytes"]))
    return starts, chunks, completions


def capture_until(websocket, capture: dict, predicate: Callable[[dict], bool]) -> dict:
    while True:
        message = websocket.receive()
        if message.get("text") is not None:
            payload = json.loads(message["text"])
            if payload["type"] == "audio_start":
                capture["starts"].append(payload)
            elif payload["type"] == "audio_chunk":
                capture["pending_chunks"].append(payload)
            elif payload["type"] == "audio_complete":
                capture["completions"].append(payload)
            if predicate(payload):
                return payload
        elif message.get("bytes") is not None:
            metadata = capture["pending_chunks"].pop(0)
            capture["chunks"].append((metadata, message["bytes"]))


def wait_until(predicate: Callable[[], bool], timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_repeated_cancellation_and_concurrent_closes_are_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    next_started = threading.Event()
    next_finished = threading.Event()
    release_next = threading.Event()
    stream_closed = threading.Event()

    def blocking_stream() -> Iterator[bytes]:
        next_started.set()
        assert release_next.wait(timeout=2)
        next_finished.set()
        try:
            yield b"discarded chunk"
        finally:
            stream_closed.set()

    class FakeWebSocket:
        close_calls = 0

        async def send_json(self, _message) -> None:
            pass

        async def send_bytes(self, _audio) -> None:
            pass

        async def close(self, code: int = 1000) -> None:
            self.close_calls += 1

    stop = Event()
    stream = blocking_stream()
    websocket = FakeWebSocket()
    monkeypatch.setattr(
        app_module,
        "stream_audio_chunks",
        lambda _text, _voice, _stop: stream,
    )
    monkeypatch.setattr(app_module, "get_sample_rate", lambda: 24_000)

    async def run_scenario() -> None:
        client = app_module.ClientConnection(websocket)
        client.start()
        job = app_module.SynthesisJob(
            id="cancelled",
            request=app_module.SpeechRequest(text="Cancel me", voice="eve"),
            client=client,
            stop=stop,
        )
        task = asyncio.create_task(app_module.stream_job_to_client(job))
        assert await asyncio.to_thread(next_started.wait, 2)

        task.cancel()
        assert await asyncio.to_thread(stop.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        release_next.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert next_finished.is_set()
        assert stream_closed.is_set()
        await asyncio.gather(
            client.close("first close"),
            client.close("second close"),
            client.close("third close"),
        )
        assert client.closed
        assert websocket.close_calls == 1

    asyncio.run(run_scenario())


def test_inflight_failure_does_not_replace_cancellation() -> None:
    next_started = threading.Event()
    release_next = threading.Event()
    stop = Event()

    def failing_stream() -> Iterator[bytes]:
        next_started.set()
        assert release_next.wait(timeout=2)
        yield from ()
        raise RuntimeError("injected in-flight failure")

    async def run_scenario() -> None:
        task = asyncio.create_task(
            app_module.await_stream_chunk(failing_stream(), stop)
        )
        assert await asyncio.to_thread(next_started.wait, 2)

        task.cancel()
        assert await asyncio.to_thread(stop.wait, 2)
        release_next.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_scenario())


def test_websocket_happy_path_preserves_fifo_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def synthesize(text: str, _voice: str) -> bytes:
        return b"RIFF" + text.encode()

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        yield f"{text}-chunk-0".encode()
        yield f"{text}-chunk-1".encode()

    with open_test_app(monkeypatch, synthesize, streamer=stream) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"id": "first", "text": "First", "voice": "eve"})
            websocket.send_json({"id": "second", "text": "Second", "voice": "eve"})

            starts, chunks, completions = collect_streams(websocket, 2)

    assert [message["id"] for message in starts] == ["first", "second"]
    assert [(metadata["id"], metadata["sequence"]) for metadata, _ in chunks] == [
        ("first", 0),
        ("first", 1),
        ("second", 0),
        ("second", 1),
    ]
    assert [chunk for _, chunk in chunks] == [
        b"First-chunk-0",
        b"First-chunk-1",
        b"Second-chunk-0",
        b"Second-chunk-1",
    ]
    assert [message["id"] for message in completions] == ["first", "second"]
    assert all(message["chunks"] == 2 for message in completions)


def test_sentence_boundary_synthesizes_before_text_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_text = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated_text.append(text)
        yield b"pcm"

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "text-stream", "voice": "eve"}
            )
            started = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            assert started["id"] == "text-stream"

            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "text-stream",
                    "text": "Hello ",
                }
            )
            first_append = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            assert first_append["characters"] == 6
            assert first_append["segments_committed"] == 0
            assert generated_text == []

            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "text-stream",
                    "text": "streaming world.",
                }
            )
            second_append = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            assert second_append["characters"] == 22
            assert second_append["segments_committed"] == 1
            wait_until(lambda: generated_text == ["Hello streaming world."])

            websocket.send_json(
                {"type": "text_finish", "id": "text-stream"}
            )
            starts, chunks, completions = collect_streams(websocket, 1)

    assert generated_text == ["Hello streaming world."]
    assert starts[0]["id"] == "text-stream"
    assert chunks[0][1] == b"pcm"
    assert completions[0]["id"] == "text-stream"


@pytest.mark.parametrize(
    ("incoming", "expected_segments"),
    [
        ("alpha beta gamma", ["alpha beta", "gamma"]),
        ("abcdefghijklmnop", ["abcdefghijkl", "mnop"]),
    ],
)
def test_size_boundary_prefers_whitespace_then_hard_splits(
    monkeypatch: pytest.MonkeyPatch,
    incoming: str,
    expected_segments: list[str],
) -> None:
    generated_text = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated_text.append(text)
        yield b"pcm"

    monkeypatch.setattr(app_module, "TEXT_SEGMENT_CHARS", 12)
    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "sized", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            websocket.send_json(
                {"type": "text_append", "id": "sized", "text": incoming}
            )
            appended = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            assert appended["segments_committed"] == 1
            wait_until(lambda: generated_text == expected_segments[:1])

            websocket.send_json({"type": "text_finish", "id": "sized"})
            starts, chunks, completions = collect_streams(websocket, 1)

    assert generated_text == expected_segments
    assert len(starts) == 1
    assert [metadata["sequence"] for metadata, _ in chunks] == [0, 1]
    assert completions[0]["chunks"] == 2


def test_long_sentence_is_split_at_segment_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_text = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated_text.append(text)
        yield b"pcm"

    monkeypatch.setattr(app_module, "TEXT_SEGMENT_CHARS", 12)
    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "long-sentence", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "long-sentence",
                    "text": "abcdefghijklmnop.",
                }
            )
            appended = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            assert appended["segments_committed"] == 2
            wait_until(lambda: generated_text[:1] == ["abcdefghijkl"])
            websocket.send_json({"type": "text_finish", "id": "long-sentence"})
            collect_streams(websocket, 1)

    assert generated_text == ["abcdefghijkl", "mnop."]
    assert all(len(segment) <= 12 for segment in generated_text)


def test_long_sentence_yields_worker_between_capped_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_text = []
    first_started = threading.Event()
    neighbor_queued = threading.Event()

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated_text.append(text)
        if text == "abcdefghijkl":
            first_started.set()
            assert neighbor_queued.wait(timeout=2)
        yield text.encode()

    monkeypatch.setattr(app_module, "TEXT_SEGMENT_CHARS", 12)
    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as backlog:
            with client.websocket_connect("/ws") as neighbor:
                backlog.send_json(
                    {
                        "type": "text_start",
                        "id": "long-sentence",
                        "voice": "eve",
                    }
                )
                receive_json_until(
                    backlog,
                    lambda message: message["type"] == "text_started",
                )
                backlog.send_json(
                    {
                        "type": "text_append",
                        "id": "long-sentence",
                        "text": "abcdefghijklmnop.",
                    }
                )
                backlog.send_json(
                    {"type": "text_finish", "id": "long-sentence"}
                )
                assert first_started.wait(timeout=2)

                neighbor.send_json(
                    {"type": "text_start", "id": "neighbor", "voice": "eve"}
                )
                neighbor.send_json(
                    {
                        "type": "text_append",
                        "id": "neighbor",
                        "text": "Other.",
                    }
                )
                neighbor.send_json({"type": "text_finish", "id": "neighbor"})
                receive_json_until(
                    neighbor,
                    lambda message: message["id"] == "neighbor"
                    and message["type"] == "queued",
                )
                neighbor_queued.set()

                wait_until(
                    lambda: generated_text
                    == ["abcdefghijkl", "Other.", "mnop."]
                )
                collect_streams(backlog, 1)
                collect_streams(neighbor, 1)

    assert generated_text == ["abcdefghijkl", "Other.", "mnop."]


def test_caught_up_stream_reenters_fifo_without_reordering_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_text = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated_text.append(text)
        yield text.encode()

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            capture = {
                "starts": [],
                "pending_chunks": [],
                "chunks": [],
                "completions": [],
            }
            websocket.send_json(
                {"type": "text_start", "id": "first-stream", "voice": "eve"}
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "first-stream",
                    "text": "First sentence.",
                }
            )
            caught_up = capture_until(
                websocket,
                capture,
                lambda message: message["type"] == "text_caught_up"
                and message["id"] == "first-stream",
            )
            assert caught_up["id"] == "first-stream"

            websocket.send_json(
                {"type": "text_start", "id": "second-stream", "voice": "eve"}
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "second-stream",
                    "text": "Second sentence.",
                }
            )
            websocket.send_json(
                {"type": "text_finish", "id": "second-stream"}
            )
            wait_until(
                lambda: generated_text
                == ["First sentence.", "Second sentence."]
            )

            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "first-stream",
                    "text": "Third sentence.",
                }
            )
            websocket.send_json(
                {"type": "text_finish", "id": "first-stream"}
            )
            capture_until(
                websocket,
                capture,
                lambda _message: len(capture["completions"]) == 2,
            )

    assert generated_text == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    assert [message["id"] for message in capture["starts"]] == [
        "first-stream",
        "second-stream",
    ]
    first_sequences = [
        metadata["sequence"]
        for metadata, _ in capture["chunks"]
        if metadata["id"] == "first-stream"
    ]
    assert first_sequences == [0, 1]
    assert {message["id"] for message in capture["completions"]} == {
        "first-stream",
        "second-stream",
    }


def test_backlogged_stream_yields_worker_after_one_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_text = []
    first_started = threading.Event()
    neighbor_queued = threading.Event()

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated_text.append(text)
        if text == "One.":
            first_started.set()
            assert neighbor_queued.wait(timeout=2)
        yield text.encode()

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as backlog:
            with client.websocket_connect("/ws") as neighbor:
                backlog.send_json(
                    {"type": "text_start", "id": "backlog", "voice": "eve"}
                )
                receive_json_until(
                    backlog,
                    lambda message: message["type"] == "text_started",
                )
                backlog.send_json(
                    {
                        "type": "text_append",
                        "id": "backlog",
                        "text": "One. Two. Three.",
                    }
                )
                backlog.send_json({"type": "text_finish", "id": "backlog"})
                assert first_started.wait(timeout=2)

                neighbor.send_json(
                    {"type": "text_start", "id": "neighbor", "voice": "eve"}
                )
                neighbor.send_json(
                    {
                        "type": "text_append",
                        "id": "neighbor",
                        "text": "Other.",
                    }
                )
                neighbor.send_json({"type": "text_finish", "id": "neighbor"})
                receive_json_until(
                    neighbor,
                    lambda message: message["id"] == "neighbor"
                    and message["type"] == "queued",
                )
                neighbor_queued.set()

                wait_until(
                    lambda: generated_text
                    == ["One.", "Other.", "Two.", "Three."]
                )
                collect_streams(backlog, 1)
                collect_streams(neighbor, 1)

    assert generated_text == ["One.", "Other.", "Two.", "Three."]


def test_idle_timeout_closes_stream_after_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "TEXT_IDLE_TIMEOUT", 0.15)
    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "idle-start", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            expired = receive_json_until(
                websocket,
                lambda message: message.get("code") == "idle_timeout",
            )
            assert expired["id"] == "idle-start"
            wait_until(
                lambda: "idle-start" not in app_module.app.state.active_job_ids
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "idle-start",
                    "text": "Too late.",
                }
            )
            unknown = receive_json_until(
                websocket,
                lambda message: message.get("code") == "unknown_text_stream",
            )
            assert unknown["id"] == "idle-start"


def test_idle_timeout_resets_on_append_and_fires_when_caught_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "TEXT_IDLE_TIMEOUT", 0.2)
    generated_text = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated_text.append(text)
        yield b"pcm"

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "idle-catchup", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "idle-catchup",
                    "text": "Hello ",
                }
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            time.sleep(0.12)
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "idle-catchup",
                    "text": "world.",
                }
            )
            capture = {
                "starts": [],
                "pending_chunks": [],
                "chunks": [],
                "completions": [],
            }
            capture_until(
                websocket,
                capture,
                lambda message: message["type"] == "text_caught_up",
            )
            assert generated_text == ["Hello world."]
            expired = capture_until(
                websocket,
                capture,
                lambda message: message.get("code") == "idle_timeout",
            )
            assert expired["id"] == "idle-catchup"
            wait_until(
                lambda: "idle-catchup"
                not in app_module.app.state.active_job_ids
            )

    assert generated_text == ["Hello world."]


def test_pending_text_chars_counts_buffer_queue_and_inflight() -> None:
    stream = app_module.TextInputStream(id="pending", voice="eve", stop=Event())
    stream.append("Hello ")
    assert stream.pending_text_chars() == 6
    stream.append("world.")
    assert stream.buffer == ""
    assert list(stream.segments) == ["Hello world."]
    leftover = app_module.TextInputStream(id="flush", voice="eve", stop=Event())
    leftover.append("no punctuation yet")
    assert leftover.buffer == "no punctuation yet"
    assert leftover.flush_buffer() == 1
    assert leftover.buffer == ""
    assert list(leftover.segments) == ["no punctuation yet"]
    assert stream.pending_text_chars() == 12
    stream.in_flight_text = stream.segments.popleft()
    assert stream.pending_text_chars() == 12
    stream.in_flight_text = None
    assert stream.pending_text_chars() == 0


def test_pending_text_limit_rejects_without_closing_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "PENDING_TEXT_CHARS", 10)
    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "pending-reject", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "pending-reject",
                    "text": "Hello ",
                }
            )
            accepted = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            assert accepted["characters"] == 6
            assert accepted["pending"] == 6
            assert accepted["limit"] == 10

            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "pending-reject",
                    "text": "world!!",
                }
            )
            rejected = receive_json_until(
                websocket,
                lambda message: message.get("code") == "pending_text_limit",
            )
            assert rejected["id"] == "pending-reject"
            assert rejected["pending"] == 5
            assert rejected["limit"] == 10
            assert rejected["rejected"] == 7
            assert "pending-reject" in app_module.app.state.active_job_ids

            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "pending-reject",
                    "text": "you.",
                }
            )
            capture = {
                "starts": [],
                "pending_chunks": [],
                "chunks": [],
                "completions": [],
            }
            retried = capture_until(
                websocket,
                capture,
                lambda message: message["type"] == "text_appended",
            )
            assert retried["characters"] == 10
            assert retried["segments_committed"] == 1


def test_append_too_large_asks_client_to_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "PENDING_TEXT_CHARS", 10)
    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "split-append", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "split-append",
                    "text": "abcdefghijk",
                }
            )
            rejected = receive_json_until(
                websocket,
                lambda message: message.get("code") == "append_too_large",
            )
            assert rejected["id"] == "split-append"
            assert rejected["pending"] == 0
            assert rejected["limit"] == 10
            assert rejected["rejected"] == 11
            assert rejected["max_append"] == 10
            assert "split-append" in app_module.app.state.active_job_ids

            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "split-append",
                    "text": "abcdefghi.",
                }
            )
            accepted = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            assert accepted["characters"] == 10
            assert accepted["pending"] == 10
            assert "split-append" in app_module.app.state.active_job_ids


def test_pending_limit_commits_unscheduled_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "PENDING_TEXT_CHARS", 20)
    started = threading.Event()
    hold = threading.Event()
    generated = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated.append(text)
        started.set()
        assert hold.wait(timeout=3)
        yield b"pcm"

    try:
        with open_test_app(
            monkeypatch,
            lambda _text, _voice: b"RIFF",
            streamer=stream,
        ) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json(
                    {"type": "text_start", "id": "flush-buffer", "voice": "eve"}
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "text_started",
                )
                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "flush-buffer",
                        "text": "unfinished buffer",
                    }
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "text_appended",
                )
                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "flush-buffer",
                        "text": " still too much",
                    }
                )
                rejected = receive_json_until(
                    websocket,
                    lambda message: message.get("code") == "pending_text_limit",
                )
                assert rejected["rejected"] == 15
                assert started.wait(timeout=2)
                assert generated == ["unfinished buffer"]
                hold.set()
                capture = {
                    "starts": [],
                    "pending_chunks": [],
                    "chunks": [],
                    "completions": [],
                }
                capture_until(
                    websocket,
                    capture,
                    lambda message: message["type"] == "text_caught_up",
                )
                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "flush-buffer",
                        "text": " still too much",
                    }
                )
                retried = capture_until(
                    websocket,
                    capture,
                    lambda message: message["type"] == "text_appended",
                )
                assert retried["characters"] == 32
                assert retried["pending"] == 15
    finally:
        hold.set()

    assert generated == ["unfinished buffer"]


def test_pending_text_limit_releases_after_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "PENDING_TEXT_CHARS", 12)
    started = threading.Event()
    hold = threading.Event()
    generated = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated.append(text)
        started.set()
        assert hold.wait(timeout=3)
        yield b"pcm"

    try:
        with open_test_app(
            monkeypatch,
            lambda _text, _voice: b"RIFF",
            streamer=stream,
        ) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json(
                    {"type": "text_start", "id": "pending-hold", "voice": "eve"}
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "text_started",
                )
                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "pending-hold",
                        "text": "Hello world.",
                    }
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "processing",
                )
                assert started.wait(timeout=2)

                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "pending-hold",
                        "text": "Next.",
                    }
                )
                rejected = receive_json_until(
                    websocket,
                    lambda message: message.get("code") == "pending_text_limit",
                )
                assert rejected["pending"] == 12
                assert rejected["rejected"] == 5
                assert "pending-hold" in app_module.app.state.active_job_ids

                hold.set()
                capture = {
                    "starts": [],
                    "pending_chunks": [],
                    "chunks": [],
                    "completions": [],
                }
                capture_until(
                    websocket,
                    capture,
                    lambda message: message["type"] == "text_caught_up",
                )

                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "pending-hold",
                        "text": "Next.",
                    }
                )
                retried = capture_until(
                    websocket,
                    capture,
                    lambda message: message["type"] == "text_appended",
                )
                assert retried["characters"] == 17
                assert retried["pending"] == 5
                websocket.send_json(
                    {"type": "text_finish", "id": "pending-hold"}
                )
                capture_until(
                    websocket,
                    capture,
                    lambda message: message["type"] == "audio_complete",
                )
    finally:
        hold.set()

    assert generated == ["Hello world.", "Next."]


def test_pending_text_limit_releases_on_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "PENDING_TEXT_CHARS", 12)
    started = threading.Event()
    hold = threading.Event()

    def stream(text: str, _voice: str, stop: Event) -> Iterator[bytes]:
        started.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if stop.is_set() or hold.is_set():
                break
            time.sleep(0.01)
        else:
            raise AssertionError("in-flight text was not cancelled")
        yield from ()

    try:
        with open_test_app(
            monkeypatch,
            lambda _text, _voice: b"RIFF",
            streamer=stream,
        ) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json(
                    {"type": "text_start", "id": "pending-drop", "voice": "eve"}
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "text_started",
                )
                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "pending-drop",
                        "text": "Hello world.",
                    }
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "processing",
                )
                assert started.wait(timeout=2)
                websocket.close()

            wait_until(
                lambda: "pending-drop" not in app_module.app.state.active_job_ids
            )

            with client.websocket_connect("/ws") as restarted:
                restarted.send_json(
                    {"type": "text_start", "id": "pending-drop", "voice": "eve"}
                )
                receive_json_until(
                    restarted,
                    lambda message: message["type"] == "text_started",
                )
                restarted.send_json(
                    {
                        "type": "text_append",
                        "id": "pending-drop",
                        "text": "Hello world.",
                    }
                )
                accepted = receive_json_until(
                    restarted,
                    lambda message: message["type"] == "text_appended",
                )
                assert accepted["pending"] == 12
                hold.set()
    finally:
        hold.set()


def test_pending_text_released_when_synthesis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "PENDING_TEXT_CHARS", 12)

    def stream(_text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        raise RuntimeError("injected synthesis failure")
        yield from ()

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "pending-fail", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "pending-fail",
                    "text": "Hello world.",
                }
            )
            receive_json_until(
                websocket,
                lambda message: message.get("type") == "error"
                and message.get("id") == "pending-fail",
            )
            wait_until(
                lambda: "pending-fail" not in app_module.app.state.active_job_ids
            )
            websocket.send_json(
                {"type": "text_start", "id": "pending-fail", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "pending-fail",
                    "text": "Hello world.",
                }
            )
            accepted = receive_json_until(
                websocket,
                lambda message: message["type"] == "text_appended",
            )
            assert accepted["pending"] == 12


def test_disconnect_releases_unfinished_text_stream_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
    ) as client:
        with client.websocket_connect("/ws") as first:
            first.send_json(
                {"type": "text_start", "id": "reusable-id", "voice": "eve"}
            )
            receive_json_until(
                first,
                lambda message: message["type"] == "text_started",
            )
            assert "reusable-id" in app_module.app.state.active_job_ids
            first.close()

        wait_until(
            lambda: "reusable-id" not in app_module.app.state.active_job_ids
        )

        with client.websocket_connect("/ws") as second:
            second.send_json(
                {"type": "text_start", "id": "reusable-id", "voice": "eve"}
            )
            restarted = receive_json_until(
                second,
                lambda message: message["type"] == "text_started",
            )
            assert restarted["id"] == "reusable-id"


def test_scheduled_stream_keeps_id_until_owning_job_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    hold = threading.Event()
    generated = []

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        generated.append(text)
        started.set()
        assert hold.wait(timeout=3)
        yield from ()

    try:
        with open_test_app(
            monkeypatch,
            lambda _text, _voice: b"RIFF",
            streamer=stream,
        ) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json(
                    {"type": "text_start", "id": "owned-id", "voice": "eve"}
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "text_started",
                )
                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "owned-id",
                        "text": "Keep this job reserved.",
                    }
                )
                receive_json_until(
                    websocket,
                    lambda message: message["type"] == "processing",
                )
                assert started.wait(timeout=2)
                owner = app_module.app.state.active_job_ids["owned-id"]

                websocket.send_json(
                    {
                        "type": "text_append",
                        "id": "owned-id",
                        "text": "x" * app_module.MAX_TEXT_LENGTH,
                    }
                )
                receive_json_until(
                    websocket,
                    lambda message: message.get("code") == "text_too_long",
                )
                assert app_module.app.state.active_job_ids["owned-id"] is owner
                hold.set()
                wait_until(
                    lambda: "owned-id" not in app_module.app.state.active_job_ids
                )

                websocket.send_json(
                    {"type": "text_start", "id": "owned-id", "voice": "eve"}
                )
                restarted = receive_json_until(
                    websocket,
                    lambda message: message["type"] == "text_started",
                )
                assert restarted["id"] == "owned-id"
    finally:
        hold.set()

    assert generated == ["Keep this job reserved."]


def test_stale_stream_cleanup_cannot_remove_new_owner() -> None:
    class FakeWebSocket:
        pass

    registry = {}
    client = app_module.ClientConnection(
        FakeWebSocket(),
        active_job_ids_registry=registry,
    )
    old_stream = app_module.TextInputStream(
        id="reused",
        voice="eve",
        stop=Event(),
    )
    new_stream = app_module.TextInputStream(
        id="reused",
        voice="eve",
        stop=Event(),
    )
    registry["reused"] = new_stream
    client.input_streams["reused"] = new_stream
    client.active_jobs["reused"] = new_stream.stop

    app_module.finalize_text_stream(
        old_stream,
        client,
        registry,
        complete_audio=False,
    )

    assert registry["reused"] is new_stream
    assert client.input_streams["reused"] is new_stream
    assert client.active_jobs["reused"] is new_stream.stop


def test_queue_rejects_jobs_over_admission_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        if text == "Active":
            started.set()
            assert release.wait(timeout=3)
        yield b"chunk"

    try:
        with open_test_app(
            monkeypatch,
            lambda _text, _voice: b"RIFF",
            streamer=stream,
            queue_size=2,
        ) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json(
                    {"id": "active", "text": "Active", "voice": "eve"}
                )
                receive_json_until(
                    websocket,
                    lambda message: message["id"] == "active"
                    and message["type"] == "processing",
                )
                assert started.wait(timeout=2)

                waiting_ids = {"waiting-1", "waiting-2", "over-limit"}
                for job_id in waiting_ids:
                    websocket.send_json(
                        {"id": job_id, "text": job_id, "voice": "eve"}
                    )

                responses = {}
                while len(responses) < len(waiting_ids):
                    message = websocket.receive_json()
                    if message.get("id") in waiting_ids:
                        responses[message["id"]] = message

                admitted = [
                    message
                    for message in responses.values()
                    if message["type"] == "queued"
                ]
                rejected = [
                    message
                    for message in responses.values()
                    if message.get("code") == "overloaded"
                ]

                assert len(admitted) == 2
                assert len(rejected) == 1
                release.set()
    finally:
        release.set()


def test_duplicate_active_job_id_is_rejected_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    calls = []

    def stream(text: str, _voice: str, stop: Event) -> Iterator[bytes]:
        calls.append(text)
        started.set()
        assert stop.wait(timeout=3)
        yield from ()

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            payload = {"id": "same-id", "text": "Original", "voice": "eve"}
            websocket.send_json(payload)
            receive_json_until(
                websocket,
                lambda message: message["id"] == "same-id"
                and message["type"] == "processing",
            )
            assert started.wait(timeout=2)

            websocket.send_json(
                {"id": "same-id", "text": "Duplicate", "voice": "eve"}
            )
            duplicate = receive_json_until(
                websocket,
                lambda message: message.get("code") == "duplicate_job_id",
            )

            assert duplicate["id"] == "same-id"
            assert app_module.app.state.jobs.qsize() == 0
            assert set(app_module.app.state.active_job_ids) == {"same-id"}
            websocket.close()
            wait_until(lambda: not app_module.app.state.active_job_ids)

    assert calls == ["Original"]


def test_disconnect_discards_clients_pending_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    disconnected = threading.Event()
    cancelled = threading.Event()
    calls = []
    closed_clients = []

    def stream(text: str, _voice: str, stop: Event) -> Iterator[bytes]:
        calls.append(text)
        if text == "Active":
            started.set()
            assert stop.wait(timeout=3)
            cancelled.set()
            return
        yield b"chunk"

    original_close = app_module.ClientConnection.close

    async def observed_close(self, reason: str) -> None:
        await original_close(self, reason)
        closed_clients.append(self)
        disconnected.set()

    monkeypatch.setattr(app_module.ClientConnection, "close", observed_close)

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"id": "active", "text": "Active", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["id"] == "active"
                and message["type"] == "processing",
            )
            assert started.wait(timeout=2)

            websocket.send_json(
                {"id": "pending", "text": "Pending", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["id"] == "pending"
                and message["type"] == "queued",
            )
            websocket.close()
            assert disconnected.wait(timeout=2)
            assert cancelled.wait(timeout=2)

            wait_until(lambda: app_module.app.state.jobs.qsize() == 0)

    assert calls == ["Active"]
    assert len(closed_clients) == 1
    assert closed_clients[0].closed
    assert closed_clients[0].active_jobs == {}
    assert closed_clients[0].input_streams == {}
    assert closed_clients[0].output_queue.empty()
    assert closed_clients[0].sender_task is None
    assert closed_clients[0].cleanup_task is None


def test_generation_failure_is_reported_and_worker_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        if text == "Fail":
            raise RuntimeError("injected synthesis failure")
        yield b"recovered-chunk"

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"id": "failure", "text": "Fail", "voice": "eve"})
            websocket.send_json(
                {"id": "recovery", "text": "Recover", "voice": "eve"}
            )

            failure = receive_json_until(
                websocket,
                lambda message: message.get("id") == "failure"
                and message["type"] == "error",
            )
            starts, chunks, completions = collect_streams(websocket, 1)

    assert failure["message"] == "injected synthesis failure"
    assert [message["id"] for message in starts][-1] == "recovery"
    assert chunks == [
        (
            {
                "type": "audio_chunk",
                "id": "recovery",
                "sequence": 0,
                "samples": len(b"recovered-chunk") // 4,
            },
            b"recovered-chunk",
        )
    ]
    assert [message["id"] for message in completions] == ["recovery"]


def test_incremental_segment_failure_releases_stream_and_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stream(text: str, _voice: str, _stop: Event) -> Iterator[bytes]:
        if text == "Fail now.":
            raise RuntimeError("injected segment failure")
        yield b"recovered"

    with open_test_app(
        monkeypatch,
        lambda _text, _voice: b"RIFF",
        streamer=stream,
    ) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "text_start", "id": "segment-failure", "voice": "eve"}
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "segment-failure",
                    "text": "Fail now.",
                }
            )
            failure = receive_json_until(
                websocket,
                lambda message: message["type"] == "error"
                and message["id"] == "segment-failure",
            )
            assert failure["message"] == "injected segment failure"
            wait_until(
                lambda: "segment-failure"
                not in app_module.app.state.active_job_ids
            )

            websocket.send_json(
                {"type": "text_start", "id": "segment-failure", "voice": "eve"}
            )
            receive_json_until(
                websocket,
                lambda message: message["type"] == "text_started"
                and message["id"] == "segment-failure",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "segment-failure",
                    "text": "Recovered.",
                }
            )
            websocket.send_json(
                {"type": "text_finish", "id": "segment-failure"}
            )
            starts, chunks, completions = collect_streams(websocket, 1)

    assert starts[-1]["id"] == "segment-failure"
    assert chunks[-1][1] == b"recovered"
    assert completions[-1]["id"] == "segment-failure"


@pytest.mark.smoke
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_TTS_TEST") != "1",
    reason="set RUN_REAL_TTS_TEST=1 to run real model inference",
)
def test_real_tts_streaming_smoke() -> None:
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/ws") as websocket:
            capture = {
                "starts": [],
                "pending_chunks": [],
                "chunks": [],
                "completions": [],
            }
            websocket.send_json(
                {
                    "type": "text_start",
                    "id": "real-smoke",
                    "voice": "eve",
                }
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "real-smoke",
                    "text": "Real incremental synthesis starts now.",
                }
            )
            capture_until(
                websocket,
                capture,
                lambda message: message["type"] == "text_caught_up",
            )
            websocket.send_json(
                {
                    "type": "text_append",
                    "id": "real-smoke",
                    "text": " It remains one ordered audio stream.",
                }
            )
            websocket.send_json(
                {"type": "text_finish", "id": "real-smoke"}
            )
            capture_until(
                websocket,
                capture,
                lambda _message: len(capture["completions"]) == 1,
            )

    assert capture["starts"] == [
        {
            "type": "audio_start",
            "id": "real-smoke",
            "sample_rate": 24_000,
            "channels": 1,
            "format": "f32le",
        }
    ]
    assert capture["chunks"]
    assert [
        metadata["sequence"] for metadata, _ in capture["chunks"]
    ] == list(
        range(len(capture["chunks"]))
    )
    assert all(len(chunk) % 4 == 0 for _, chunk in capture["chunks"])
    assert capture["completions"][0]["chunks"] == len(capture["chunks"])
    assert capture["completions"][0]["samples"] == sum(
        len(chunk) // 4 for _, chunk in capture["chunks"]
    )


def test_home_page_loads_javascript_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    with open_test_app(monkeypatch, lambda text, voice: b"RIFF") as client:
        home = client.get("/")
        sdk = client.get("/sdk/tts-client.js")

    assert home.status_code == 200
    assert 'import { TtsClient } from "/sdk/tts-client.js"' in home.text
    assert sdk.status_code == 200
    assert "startStream" in sdk.text
    assert "appendText" in sdk.text
    assert "finishText" in sdk.text
    assert "export class TtsClient" in sdk.text
