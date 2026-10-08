"""Live WebSocket harness for the TTS server.

Usage:
    python harness.py --url http://127.0.0.1:8000
    python harness.py --url http://127.0.0.1:8000 --clients 8 --burst-extra 4 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import websockets
from websockets.exceptions import ConnectionClosed


SENTENCES = (
    "The first sentence should start synthesis as soon as the period arrives.",
    "This second sentence keeps the same stream going after the worker catches up.",
    "The final sentence confirms the client still receives one ordered audio stream.",
)
SHORT_SENTENCE = "Short burst sentence number {n}."


def repeated_sentences(repeats: int) -> tuple[str, ...]:
    return SENTENCES * max(1, repeats)


def websocket_url(base: str) -> str:
    parsed = urlparse(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse(parsed._replace(scheme=scheme, path="/ws", query="", fragment=""))


def health_url(base: str) -> str:
    return urljoin(base.rstrip("/") + "/", "health")


def now() -> float:
    return time.perf_counter()


def ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


def split_text(text: str, max_size: int) -> list[str]:
    limit = max(1, max_size)
    if len(text) <= limit:
        if len(text) <= 1:
            return [text]
        mid = (len(text) + 1) // 2
        return [text[:mid], text[mid:]]
    pieces: list[str] = []
    offset = 0
    while offset < len(text):
        end = min(len(text), offset + limit)
        if end < len(text):
            window = text[offset:end]
            break_at = window.rfind(" ")
            if break_at >= 1:
                end = offset + break_at + 1
        if end <= offset:
            end = min(len(text), offset + limit)
        pieces.append(text[offset:end])
        offset = end
    return pieces


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": round(statistics.fmean(values), 1),
        "p95_ms": round(statistics.quantiles(values, n=20)[18], 1)
        if len(values) >= 20
        else max(values),
    }


@dataclass
class StreamMetrics:
    stream_id: str
    first_text_at: float | None = None
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    chunk_gaps_ms: list[float] = field(default_factory=list)
    binary_chunks: int = 0
    declared_samples: int = 0
    next_sequence: int = 0
    audio_started: bool = False
    audio_completed: bool = False
    sequences: list[int] = field(default_factory=list)
    admitted: bool = False
    overloads: int = 0
    failures: list[str] = field(default_factory=list)
    disconnected: bool = False
    append_events: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    last_max_append: int | None = None
    sample_rate: int | None = None
    playback_buffered_until: float | None = None
    stall_count: int = 0
    stalled_s: float = 0.0
    longest_stall_s: float = 0.0
    playback_closed: bool = False

    @property
    def time_to_first_audio_ms(self) -> float | None:
        if self.first_text_at is None or self.first_audio_at is None:
            return None
        return ms(self.first_audio_at - self.first_text_at)

    @property
    def total_stalled_ms(self) -> float:
        return ms(self.stalled_s) or 0.0

    @property
    def longest_stall_ms(self) -> float:
        return ms(self.longest_stall_s) or 0.0

    def _record_stall(self, stall: float) -> None:
        if stall <= 0:
            return
        self.stall_count += 1
        self.stalled_s += stall
        if stall > self.longest_stall_s:
            self.longest_stall_s = stall

    def note_virtual_chunk(self, received: float, samples: int) -> None:
        rate = self.sample_rate if self.sample_rate and self.sample_rate > 0 else 24_000
        duration = samples / rate
        if self.playback_buffered_until is None:
            # Startup wait before the first chunk is not a stall.
            self.playback_buffered_until = received + duration
            return
        if received > self.playback_buffered_until:
            self._record_stall(received - self.playback_buffered_until)
            self.playback_buffered_until = received + duration
            return
        self.playback_buffered_until += duration

    def stop_virtual_playback(self) -> None:
        """Stop measuring without treating remaining silence as a stall."""
        self.playback_closed = True

    def close_virtual_playback(self, at: float | None = None) -> None:
        if self.playback_closed:
            return
        self.playback_closed = True
        ended = now() if at is None else at
        if self.playback_buffered_until is None:
            return
        # Buffer ran out and no later chunk arrived before the stream ended.
        self._record_stall(ended - self.playback_buffered_until)


def stall_summary(streams: list[StreamMetrics]) -> dict[str, float]:
    return {
        "count": sum(stream.stall_count for stream in streams),
        "total_ms": ms(sum(stream.stalled_s for stream in streams)) or 0.0,
        "longest_ms": ms(
            max((stream.longest_stall_s for stream in streams), default=0.0)
        )
        or 0.0,
    }


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    notes: list[str] = field(default_factory=list)
    streams: list[StreamMetrics] = field(default_factory=list)
    overloads: int = 0
    failures: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, object]:
        first_audio = [
            stream.time_to_first_audio_ms
            for stream in self.streams
            if stream.time_to_first_audio_ms is not None
        ]
        gaps = [
            gap
            for stream in self.streams
            for gap in stream.chunk_gaps_ms
        ]
        return {
            "name": self.name,
            "ok": self.ok,
            "notes": self.notes,
            "overloads": self.overloads,
            "failures": self.failures,
            "streams": len(self.streams),
            "completed_audio": sum(stream.audio_completed for stream in self.streams),
            "time_to_first_audio": summarize(first_audio),
            "chunk_gaps": summarize(gaps),
            "playback_stalls": stall_summary(self.streams),
            "elapsed_s": round(self.elapsed_s, 2),
        }


class TtsClient:
    def __init__(self, url: str, name: str) -> None:
        self.url = url
        self.name = name
        self.socket = None
        self.streams: dict[str, StreamMetrics] = {}
        self.unexpected: list[str] = []
        self._pending_chunk: tuple[StreamMetrics, int, int] | None = None
        self._reader_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        self.socket = await websockets.connect(self.url, max_size=None)
        # Always drain inbound frames. The server closes a client whose
        # 10-deep output queue fills, and first-audio timing is only valid
        # if we read while later text is still being sent.
        self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        if self.socket is not None:
            await self.socket.close()
            self.socket = None
        if self._reader_task is not None:
            with suppress(asyncio.CancelledError, ConnectionClosed):
                await self._reader_task
            self._reader_task = None

    async def _read_loop(self) -> None:
        try:
            async for message in self.socket:
                self.handle_message(message)
        except ConnectionClosed:
            pass
        finally:
            for metrics in self.streams.values():
                metrics.disconnected = True

    async def start_stream(self, stream_id: str, voice: str = "eve") -> StreamMetrics:
        metrics = StreamMetrics(stream_id=stream_id)
        self.streams[stream_id] = metrics
        await self.socket.send(
            json.dumps({"type": "text_start", "id": stream_id, "voice": voice})
        )
        return metrics

    async def append(self, stream_id: str, text: str) -> None:
        metrics = self.streams[stream_id]
        if metrics.first_text_at is None:
            metrics.first_text_at = now()
        pending = [text]
        while pending:
            current = pending[0]
            if metrics.overloads or metrics.failures:
                return
            await self.socket.send(
                json.dumps({"type": "text_append", "id": stream_id, "text": current})
            )
            try:
                result = await asyncio.wait_for(metrics.append_events.get(), timeout=15)
            except TimeoutError:
                return
            if result == "text_appended":
                pending.pop(0)
                continue
            if result == "append_too_large":
                limit = metrics.last_max_append or max(1, len(current) // 2)
                pieces = split_text(current, limit)
                if pieces == [current]:
                    metrics.failures.append("append_too_large could not be split")
                    return
                pending[:1] = pieces
                continue
            if result != "pending_text_limit":
                return
            await asyncio.sleep(0.05)

    async def finish(self, stream_id: str) -> None:
        await self.socket.send(json.dumps({"type": "text_finish", "id": stream_id}))

    async def send_incremental(self, stream_id: str, parts: tuple[str, ...]) -> StreamMetrics:
        metrics = await self.start_stream(stream_id)
        for part in parts:
            if metrics.overloads or metrics.failures:
                return metrics
            await self.append(stream_id, part + " ")
            await asyncio.sleep(0.02)
        if not metrics.overloads and not metrics.failures:
            await self.finish(stream_id)
        return metrics

    def _record_unexpected(self, detail: str) -> None:
        if self.streams:
            for stream in self.streams.values():
                if detail not in stream.failures:
                    stream.failures.append(detail)
            return
        if detail not in self.unexpected:
            self.unexpected.append(detail)

    def _reject_metadata(
        self,
        metrics: StreamMetrics,
        detail: str,
    ) -> None:
        metrics.failures.append(detail)
        self._pending_chunk = None

    def _validate_audio_chunk(
        self,
        metrics: StreamMetrics,
        payload: dict[str, object],
    ) -> tuple[int, int] | None:
        expected = {"type", "id", "sequence", "samples"}
        missing = expected - set(payload)
        extra = set(payload) - expected
        if missing:
            self._reject_metadata(
                metrics,
                f"audio_chunk missing fields {sorted(missing)}",
            )
            return None
        if extra:
            self._reject_metadata(
                metrics,
                f"audio_chunk unexpected fields {sorted(extra)}",
            )
            return None
        if payload["id"] != metrics.stream_id:
            self._reject_metadata(
                metrics,
                f"audio_chunk id {payload['id']!r} does not match {metrics.stream_id}",
            )
            return None
        sequence = payload["sequence"]
        samples = payload["samples"]
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            self._reject_metadata(
                metrics,
                f"audio_chunk sequence {sequence!r} is not an int",
            )
            return None
        if sequence != metrics.next_sequence:
            self._reject_metadata(
                metrics,
                f"audio_chunk sequence {sequence} expected {metrics.next_sequence}",
            )
            return None
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            self._reject_metadata(
                metrics,
                f"audio_chunk samples {samples!r} is not a positive int",
            )
            return None
        return sequence, samples

    def _owned_stream(self, payload: dict[str, object]) -> StreamMetrics | None:
        stream_id = payload.get("id")
        metrics = self.streams.get(stream_id) if stream_id else None
        if metrics is not None:
            return metrics
        self._record_unexpected(
            f"unexpected {payload.get('type')} for stream {stream_id!r}; "
            f"harness streams are {sorted(self.streams)}"
        )
        return None

    def handle_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            pending = self._pending_chunk
            self._pending_chunk = None
            if pending is None:
                self._record_unexpected(
                    "binary audio arrived without an audio_chunk for a harness stream"
                )
                return
            metrics, sequence, samples = pending
            if not metrics.audio_started:
                metrics.failures.append(
                    f"binary audio for {metrics.stream_id} before audio_start"
                )
                return
            if len(message) % 4 != 0:
                metrics.failures.append(
                    f"audio_chunk {sequence} binary length {len(message)} "
                    "is not float32 aligned"
                )
                return
            pcm_samples = len(message) // 4
            if pcm_samples != samples:
                metrics.failures.append(
                    f"audio_chunk {sequence} samples={samples} "
                    f"but binary has {pcm_samples} samples"
                )
                return
            received = now()
            if metrics.first_audio_at is None:
                metrics.first_audio_at = received
            elif metrics.last_audio_at is not None:
                metrics.chunk_gaps_ms.append(ms(received - metrics.last_audio_at))
            metrics.last_audio_at = received
            metrics.binary_chunks += 1
            metrics.declared_samples += samples
            metrics.next_sequence += 1
            metrics.note_virtual_chunk(received, samples)
            return

        payload = json.loads(message)
        message_type = payload.get("type")
        if message_type in {"audio_start", "audio_chunk", "audio_complete"}:
            metrics = self._owned_stream(payload)
            if metrics is None:
                return
            if message_type == "audio_start":
                if metrics.audio_started:
                    metrics.failures.append(
                        f"duplicate audio_start for {metrics.stream_id}"
                    )
                    return
                metrics.audio_started = True
                metrics.admitted = True
                rate = payload.get("sample_rate")
                if isinstance(rate, int) and not isinstance(rate, bool) and rate > 0:
                    metrics.sample_rate = rate
            elif message_type == "audio_chunk":
                if not metrics.audio_started:
                    metrics.failures.append(
                        f"audio_chunk for {metrics.stream_id} before audio_start"
                    )
                    return
                if self._pending_chunk is not None:
                    metrics.failures.append(
                        "audio_chunk arrived before the previous chunk's binary payload"
                    )
                    self._pending_chunk = None
                    return
                validated = self._validate_audio_chunk(metrics, payload)
                if validated is None:
                    return
                sequence, samples = validated
                metrics.sequences.append(sequence)
                self._pending_chunk = (metrics, sequence, samples)
            else:
                if not metrics.audio_started:
                    metrics.failures.append(
                        f"audio_complete for {metrics.stream_id} before audio_start"
                    )
                    return
                if self._pending_chunk is not None:
                    metrics.failures.append(
                        "audio_complete arrived before the last chunk's binary payload"
                    )
                    self._pending_chunk = None
                    return
                chunks = payload.get("chunks")
                samples = payload.get("samples")
                if chunks != metrics.binary_chunks:
                    metrics.failures.append(
                        f"audio_complete chunks={chunks} "
                        f"expected {metrics.binary_chunks}"
                    )
                    return
                if samples != metrics.declared_samples:
                    metrics.failures.append(
                        f"audio_complete samples={samples} "
                        f"expected {metrics.declared_samples}"
                    )
                    return
                metrics.audio_completed = True
                metrics.close_virtual_playback()
            return

        if message_type in {"queued", "processing"}:
            metrics = self.streams.get(payload.get("id"))
            if metrics is not None and not metrics.overloads:
                metrics.admitted = True
            return

        if message_type == "text_appended":
            metrics = self.streams.get(payload.get("id"))
            if metrics is not None:
                metrics.append_events.put_nowait("text_appended")
            return

        if message_type != "error":
            return
        metrics = self.streams.get(payload.get("id"))
        if metrics is None:
            return
        code = payload.get("code")
        detail = payload.get("message", "unknown error")
        if code == "overloaded":
            metrics.overloads += 1
        elif code == "pending_text_limit":
            metrics.append_events.put_nowait("pending_text_limit")
        elif code == "append_too_large":
            max_append = payload.get("max_append", payload.get("limit"))
            if isinstance(max_append, int) and not isinstance(max_append, bool):
                metrics.last_max_append = max_append
            metrics.append_events.put_nowait("append_too_large")
        elif metrics.overloads:
            return
        else:
            metrics.failures.append(f"{code or 'error'}: {detail}")
            metrics.close_virtual_playback()

    async def drain_until(
        self,
        predicate,
        timeout: float = 20,
    ) -> None:
        deadline = now() + timeout
        while now() < deadline:
            if predicate():
                return
            if self._reader_task is not None and self._reader_task.done():
                return
            await asyncio.sleep(0.01)

    async def wait_for_first_audio(self, stream_id: str, timeout: float = 20) -> None:
        await self.drain_until(
            lambda: self.streams[stream_id].first_audio_at is not None
            or self.streams[stream_id].overloads
            or self.streams[stream_id].failures,
            timeout=timeout,
        )

    async def wait_for_complete(self, stream_id: str, timeout: float = 30) -> None:
        await self.drain_until(
            lambda: self.streams[stream_id].audio_completed
            or self.streams[stream_id].overloads
            or self.streams[stream_id].failures,
            timeout=timeout,
        )


def owned_stream_ok(stream: StreamMetrics) -> bool:
    return (
        stream.audio_started
        and stream.audio_completed
        and stream.binary_chunks > 0
        and stream.sequences == list(range(stream.binary_chunks))
        and not stream.failures
    )


def collect(result: ScenarioResult, clients: list[TtsClient]) -> None:
    for client in clients:
        for stream in client.streams.values():
            stream.close_virtual_playback()
        result.streams.extend(client.streams.values())
        result.overloads += sum(stream.overloads for stream in client.streams.values())
        result.failures.extend(
            f"{client.name}: {failure}" for failure in client.unexpected
        )
        for stream in client.streams.values():
            if stream.overloads:
                continue
            result.failures.extend(
                f"{client.name}/{stream.stream_id}: {failure}"
                for failure in stream.failures
            )


async def fetch_health(base: str) -> dict[str, str]:
    async with httpx.AsyncClient() as http:
        response = await http.get(health_url(base), timeout=5)
        response.raise_for_status()
        return response.json()


async def fetch_capacity(base: str) -> int:
    return int((await fetch_health(base))["capacity"])


async def wait_for_idle(base: str, timeout: float = 20) -> dict[str, str] | None:
    deadline = now() + timeout
    latest: dict[str, str] = {}
    while now() < deadline:
        latest = await fetch_health(base)
        if int(latest.get("queued", "1")) == 0 and int(latest.get("active_jobs", "1")) == 0:
            return latest
        await asyncio.sleep(0.05)
    return latest or None


async def scenario_single_client(url: str, repeats: int = 1) -> ScenarioResult:
    result = ScenarioResult(name="single_client_streaming_text", ok=False)
    parts = repeated_sentences(repeats)
    client = TtsClient(url, "single")
    await client.connect()
    try:
        stream_id = str(uuid.uuid4())
        await client.start_stream(stream_id)
        first, *rest = parts
        await client.append(stream_id, first + " ")
        await client.wait_for_first_audio(stream_id)
        stream = client.streams[stream_id]
        started_early = (
            stream.audio_started
            and stream.first_audio_at is not None
            and not stream.audio_completed
        )
        if not started_early:
            collect(result, [client])
            result.notes.append(
                "audio did not start from the first text part before later parts were sent"
            )
            return result

        chunks_before_rest = stream.binary_chunks
        early_ms = stream.time_to_first_audio_ms
        for part in rest:
            await client.append(stream_id, part + " ")
            await asyncio.sleep(0.02)
        await client.finish(stream_id)
        await client.wait_for_complete(stream_id, timeout=max(30, 25 * repeats))
        collect(result, [client])
        if owned_stream_ok(stream) and started_early and not result.failures:
            result.ok = True
            result.notes.append(
                f"first audio after first part in {early_ms} ms "
                f"({chunks_before_rest} chunks before remaining text); "
                f"{len(parts)} parts ({repeats}x); "
                f"{stream.binary_chunks} chunks total in sequences "
                f"{stream.sequences[:8]}..."
            )
        else:
            result.notes.append("stream did not complete with binary audio")
    finally:
        await client.close()
    return result


async def scenario_concurrent_clients(
    url: str,
    client_count: int,
    repeats: int = 1,
) -> ScenarioResult:
    result = ScenarioResult(name="multiple_concurrent_text", ok=False)
    parts = repeated_sentences(repeats)
    clients = [TtsClient(url, f"client-{index}") for index in range(client_count)]
    await asyncio.gather(*(client.connect() for client in clients))
    try:
        async def run_client(index: int, client: TtsClient) -> None:
            stream_id = f"concurrent-{index}-{uuid.uuid4()}"
            await client.send_incremental(stream_id, parts)
            await client.wait_for_complete(stream_id, timeout=max(30, 25 * repeats))

        await asyncio.gather(
            *(run_client(index, client) for index, client in enumerate(clients))
        )
        collect(result, clients)
        completed = [stream for stream in result.streams if owned_stream_ok(stream)]
        if len(completed) == len(clients) and not result.failures:
            result.ok = True
            result.notes.append(
                f"{len(completed)} clients completed independent streams "
                f"({len(parts)} parts, {repeats}x)"
            )
        else:
            result.notes.append(
                f"completed {len(completed)}/{len(clients)}; failures={len(result.failures)}"
            )
    finally:
        await asyncio.gather(*(client.close() for client in clients))
    return result


async def scenario_admission_burst(
    url: str,
    base: str,
    capacity: int,
    burst_extra: int,
) -> ScenarioResult:
    result = ScenarioResult(name="burst_exceeds_admission_capacity", ok=False)
    burst = capacity + burst_extra
    clients = [TtsClient(url, f"burst-{index}") for index in range(burst)]
    recovery = TtsClient(url, "resume")
    await asyncio.gather(*(client.connect() for client in clients))
    try:
        async def fire(index: int, client: TtsClient) -> None:
            stream_id = f"burst-{index}-{uuid.uuid4()}"
            await client.send_incremental(stream_id, (SHORT_SENTENCE.format(n=index),))

        await asyncio.gather(
            *(fire(index, client) for index, client in enumerate(clients))
        )
        await asyncio.gather(
            *(
                client.drain_until(
                    lambda current=client: any(
                        stream.overloads
                        or stream.failures
                        or owned_stream_ok(stream)
                        for stream in current.streams.values()
                    ),
                    timeout=30,
                )
                for client in clients
            )
        )
        collect(result, clients)
        admitted = [
            stream
            for stream in result.streams
            if stream.admitted and not stream.overloads
        ]
        rejected = [stream for stream in result.streams if stream.overloads]
        inflight_ok = [stream for stream in admitted if owned_stream_ok(stream)]
        unfinished = [
            stream
            for stream in result.streams
            if not stream.overloads and stream not in inflight_ok
        ]
        burst_ok = False
        if result.overloads == 0:
            result.notes.append(
                f"admitted {len(admitted)}/{burst} with no overloads; "
                "increase burst or lower capacity"
            )
        elif unfinished:
            result.notes.append(
                f"in-flight requests did not all succeed: "
                f"{len(inflight_ok)}/{len(admitted)} completed, "
                f"{len(unfinished)} unfinished, overloads={len(rejected)}"
            )
            result.failures.extend(
                f"inflight/{stream.stream_id}: "
                f"admitted={stream.admitted} complete={stream.audio_completed} "
                f"chunks={stream.binary_chunks} failures={stream.failures}"
                for stream in unfinished
            )
        elif result.failures:
            result.notes.append("burst produced real failures in addition to overloads")
        else:
            burst_ok = True
            result.notes.append(
                f"capacity={capacity}; in-flight {len(inflight_ok)} succeeded; "
                f"overloads={len(rejected)}; overloads counted separately from failures"
            )

        idle = await wait_for_idle(base)
        if idle is None or int(idle.get("queued", "1")) or int(idle.get("active_jobs", "1")):
            result.notes.append(
                f"server did not drain after burst: queued={idle.get('queued') if idle else '?'} "
                f"active_jobs={idle.get('active_jobs') if idle else '?'}"
            )
            return result

        await recovery.connect()
        resume_id = f"resume-{uuid.uuid4()}"
        await recovery.send_incremental(
            resume_id,
            (SHORT_SENTENCE.format(n="resume"),),
        )
        await recovery.wait_for_complete(resume_id)
        collect(result, [recovery])
        resume = recovery.streams[resume_id]
        if owned_stream_ok(resume) and not resume.overloads:
            result.notes.append(
                "resumed a full stream after burst drained "
                f"(queued={idle['queued']}, active_jobs={idle['active_jobs']})"
            )
            result.ok = burst_ok
        else:
            result.notes.append(
                "post-burst stream did not complete: "
                f"chunks={resume.binary_chunks} complete={resume.audio_completed} "
                f"overloads={resume.overloads} failures={resume.failures}"
            )
    finally:
        await asyncio.gather(
            *(client.close() for client in clients),
            recovery.close(),
        )
    return result


async def scenario_disconnect_mid_audio(url: str) -> ScenarioResult:
    result = ScenarioResult(name="disconnect_after_first_chunk", ok=False)
    dropping = TtsClient(url, "dropping")
    surviving = TtsClient(url, "surviving")
    await asyncio.gather(dropping.connect(), surviving.connect())
    try:
        drop_id = f"drop-{uuid.uuid4()}"
        live_id = f"live-{uuid.uuid4()}"
        await asyncio.gather(
            dropping.send_incremental(drop_id, SENTENCES),
            surviving.send_incremental(live_id, SENTENCES),
        )
        await dropping.wait_for_first_audio(drop_id)
        drop_stream = dropping.streams[drop_id]
        if drop_stream.first_audio_at is None or not drop_stream.audio_started:
            result.notes.append("dropping client never received binary audio")
            collect(result, [dropping, surviving])
            return result

        for stream in dropping.streams.values():
            stream.stop_virtual_playback()
        await dropping.close()
        drop_stream.disconnected = True
        await surviving.wait_for_complete(live_id)
        collect(result, [dropping, surviving])
        live = surviving.streams[live_id]
        if owned_stream_ok(live) and not result.failures:
            result.ok = True
            result.notes.append(
                "dropping client left after first binary chunk; "
                f"surviving client completed {live.binary_chunks} chunks"
            )
        else:
            result.notes.append(
                "surviving client did not complete after peer disconnect: "
                f"chunks={live.binary_chunks} complete={live.audio_completed} "
                f"disconnected={live.disconnected} failures={live.failures}"
            )
    finally:
        await surviving.close()
        await dropping.close()
    return result


def print_report(
    base: str,
    capacity: int,
    results: list[ScenarioResult],
    *,
    clients: int,
    burst_extra: int,
    repeats: int,
) -> None:
    report = {
        "target": base,
        "queue_capacity": capacity,
        "clients": clients,
        "burst_extra": burst_extra,
        "repeats": repeats,
        "scenarios": [result.as_dict() for result in results],
        "totals": {
            "ok": all(result.ok for result in results),
            "overloads": sum(result.overloads for result in results),
            "failures": [
                failure for result in results for failure in result.failures
            ],
            "time_to_first_audio": summarize(
                [
                    stream.time_to_first_audio_ms
                    for result in results
                    for stream in result.streams
                    if stream.time_to_first_audio_ms is not None
                ]
            ),
            "chunk_gaps": summarize(
                [
                    gap
                    for result in results
                    for stream in result.streams
                    for gap in stream.chunk_gaps_ms
                ]
            ),
            "playback_stalls": stall_summary(
                [stream for result in results for stream in result.streams]
            ),
        },
    }
    print(json.dumps(report, indent=2))
    print()
    print("=== TTS harness summary ===")
    print(
        f"target: {base}  queue_capacity: {capacity}  "
        f"clients: {clients}  burst_extra: {burst_extra}  repeats: {repeats}"
    )
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        summary = result.as_dict()
        first = summary["time_to_first_audio"]
        gaps = summary["chunk_gaps"]
        stalls = summary["playback_stalls"]
        first_text = (
            f"ttfa {first['min_ms']}-{first['max_ms']} ms (mean {first['mean_ms']})"
            if first
            else "ttfa n/a"
        )
        gap_text = (
            f"chunk gaps {gaps['min_ms']}-{gaps['max_ms']} ms "
            f"(mean {gaps['mean_ms']}, p95 {gaps['p95_ms']})"
            if gaps
            else "chunk gaps n/a"
        )
        stall_text = (
            f"stalls {stalls['count']} total {stalls['total_ms']} ms "
            f"longest {stalls['longest_ms']} ms"
        )
        print(
            f"  [{status}] {result.name}: {result.elapsed_s:.1f}s "
            f"overloads={result.overloads} failures={len(result.failures)} "
            f"{first_text}; {gap_text}; {stall_text}"
        )
        for note in result.notes:
            print(f"         {note}")
    totals = report["totals"]
    overall = "PASS" if totals["ok"] else "FAIL"
    stalls = totals["playback_stalls"]
    print(
        f"overall: {overall}  overloads={totals['overloads']}  "
        f"real_failures={len(totals['failures'])}  "
        f"stalls={stalls['count']} total {stalls['total_ms']} ms "
        f"longest {stalls['longest_ms']} ms"
    )


async def run_scenario(label: str, factory) -> ScenarioResult:
    print(f"starting {label}...", flush=True)
    started = now()
    result = await factory()
    result.elapsed_s = now() - started
    status = "PASS" if result.ok else "FAIL"
    print(
        f"completed {result.name} [{status}] in {result.elapsed_s:.1f}s",
        flush=True,
    )
    return result


async def run(base: str, clients: int, burst_extra: int, repeats: int) -> int:
    url = websocket_url(base)
    capacity = await fetch_capacity(base)
    results = [
        await run_scenario(
            "single_client_streaming_text",
            lambda: scenario_single_client(url, repeats),
        ),
        await run_scenario(
            "multiple_concurrent_text",
            lambda: scenario_concurrent_clients(url, clients, repeats),
        ),
        await run_scenario(
            "burst_exceeds_admission_capacity",
            lambda: scenario_admission_burst(url, base, capacity, burst_extra),
        ),
        await run_scenario(
            "disconnect_after_first_chunk",
            lambda: scenario_disconnect_mid_audio(url),
        ),
    ]
    print_report(
        base,
        capacity,
        results,
        clients=clients,
        burst_extra=burst_extra,
        repeats=repeats,
    )
    return 0 if all(result.ok for result in results) else 1


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live TTS WebSocket scenarios.")
    parser.add_argument(
        "--url",
        required=True,
        help="HTTP origin of the TTS server, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--clients",
        type=positive_int,
        default=3,
        help="Concurrent streaming clients for multiple_concurrent_text (default: 3)",
    )
    parser.add_argument(
        "--burst-extra",
        type=positive_int,
        default=6,
        help="Extra clients above queue capacity for the admission burst (default: 6)",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=1,
        help="Repeat the input text this many times in single-client and concurrent scenarios (default: 1)",
    )
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            run(args.url.rstrip("/"), args.clients, args.burst_extra, args.repeats)
        )
    )


if __name__ == "__main__":
    main()
