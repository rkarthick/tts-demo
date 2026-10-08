"""Queued local web UI and API for Pocket TTS."""

import asyncio
import logging
import os
import re
from collections import deque
from collections.abc import Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator

from tts import (
    ENGLISH_VOICES,
    generate_wav,
    get_sample_rate,
    get_tts,
    stream_audio_chunks,
)

logger = logging.getLogger(__name__)
ActiveJobOwners = dict[str, object]
MAX_QUEUE_SIZE = max(1, int(os.environ.get("TTS_MAX_QUEUE_SIZE", "10")))
CLIENT_OUTPUT_QUEUE_SIZE = max(
    1, int(os.environ.get("TTS_CLIENT_OUTPUT_QUEUE_SIZE", "10"))
)
CLIENT_SEND_TIMEOUT = max(
    0.1, float(os.environ.get("TTS_CLIENT_SEND_TIMEOUT", "5"))
)
MAX_TEXT_LENGTH = max(1, int(os.environ.get("TTS_MAX_TEXT_LENGTH", "50000")))
MAX_CLIENT_INPUT_STREAMS = max(
    1, int(os.environ.get("TTS_CLIENT_INPUT_STREAMS", "10"))
)
TEXT_SEGMENT_CHARS = max(
    32, int(os.environ.get("TTS_TEXT_SEGMENT_CHARS", "80"))
)
TEXT_IDLE_TIMEOUT = max(
    0.5, float(os.environ.get("TTS_TEXT_IDLE_TIMEOUT", "20"))
)
PENDING_TEXT_CHARS = max(
    TEXT_SEGMENT_CHARS,
    int(os.environ.get("TTS_PENDING_TEXT_CHARS", str(TEXT_SEGMENT_CHARS * 5))),
)
SENTENCE_BOUNDARY = re.compile(r"""[.!?](?:["')\]]*)?(?:\s+|$)""")


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    voice: str = "eve"

    @field_validator("text")
    @classmethod
    def text_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value.strip()

    @field_validator("voice")
    @classmethod
    def voice_must_exist(cls, value: str) -> str:
        if value not in ENGLISH_VOICES:
            raise ValueError("unknown English voice")
        return value


class StreamIdentifier(BaseModel):
    id: str = Field(min_length=1, max_length=128)


class TextStart(StreamIdentifier):
    voice: str = "eve"

    @field_validator("voice")
    @classmethod
    def voice_must_exist(cls, value: str) -> str:
        if value not in ENGLISH_VOICES:
            raise ValueError("unknown English voice")
        return value


class TextAppend(StreamIdentifier):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class TextFinish(StreamIdentifier):
    pass


@dataclass
class TextInputStream:
    id: str
    voice: str
    stop: Event
    buffer: str = ""
    segments: deque[str] = field(default_factory=deque)
    characters: int = 0
    has_text: bool = False
    scheduled: bool = False
    finished: bool = False
    audio_started: bool = False
    next_sequence: int = 0
    total_samples: int = 0
    in_flight_text: str | None = None
    idle_handle: asyncio.TimerHandle | None = field(default=None, init=False)
    idle_callback: object | None = field(default=None, init=False)

    def pending_text_chars(self) -> int:
        inflight = len(self.in_flight_text) if self.in_flight_text else 0
        return len(self.buffer) + sum(len(segment) for segment in self.segments) + inflight

    def append(self, text: str) -> int:
        if self.finished:
            raise ValueError("text stream is already finished")
        if self.characters + len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"text exceeds {MAX_TEXT_LENGTH} characters")
        self.buffer += text
        self.characters += len(text)
        self.has_text = self.has_text or bool(text.strip())
        return self._commit_boundaries()

    def flush_buffer(self) -> int:
        committed = self._commit_boundaries()
        remainder = self.buffer.strip()
        self.buffer = ""
        if remainder:
            self.segments.append(remainder)
            committed += 1
        return committed

    def finish(self) -> int:
        if self.finished:
            raise ValueError("text stream is already finished")
        self.finished = True
        committed = self.flush_buffer()
        if not self.has_text:
            raise ValueError("text stream cannot be empty")
        return committed

    def _commit_boundaries(self) -> int:
        committed = 0
        while self.buffer:
            if len(self.buffer) >= TEXT_SEGMENT_CHARS:
                prefix = self.buffer[:TEXT_SEGMENT_CHARS]
                sentence = SENTENCE_BOUNDARY.search(prefix)
                if sentence is not None:
                    committed += self._commit_prefix(sentence.end())
                    continue
                whitespace = list(re.finditer(r"\s+", prefix))
                split_at = whitespace[-1].end() if whitespace else TEXT_SEGMENT_CHARS
                committed += self._commit_prefix(split_at)
                continue

            sentence = SENTENCE_BOUNDARY.search(self.buffer)
            if sentence is None:
                break
            committed += self._commit_prefix(sentence.end())
        return committed

    def _commit_prefix(self, end: int) -> int:
        segment = self.buffer[:end].strip()
        self.buffer = self.buffer[end:]
        if not segment:
            return 0
        self.segments.append(segment)
        return 1


def cancel_idle_timer(text_stream: TextInputStream) -> None:
    handle = text_stream.idle_handle
    if handle is not None:
        handle.cancel()
        text_stream.idle_handle = None


def arm_idle_timer(text_stream: TextInputStream) -> None:
    cancel_idle_timer(text_stream)
    callback = text_stream.idle_callback
    if (
        callback is None
        or text_stream.finished
        or text_stream.stop.is_set()
    ):
        return
    text_stream.idle_handle = asyncio.get_running_loop().call_later(
        TEXT_IDLE_TIMEOUT,
        callback,
    )


@dataclass
class OutboundMessage:
    metadata: dict[str, object]
    audio: bytes | None = None


@dataclass
class ClientConnection:
    websocket: WebSocket
    active_job_ids_registry: ActiveJobOwners | None = None
    active: bool = True
    output_queue: asyncio.Queue[OutboundMessage] = field(init=False)
    sender_task: asyncio.Task[None] | None = field(default=None, init=False)
    cleanup_task: asyncio.Task[None] | None = field(default=None, init=False)
    active_jobs: dict[str, Event] = field(default_factory=dict, init=False)
    input_streams: dict[str, TextInputStream] = field(
        default_factory=dict,
        init=False,
    )
    closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.output_queue = asyncio.Queue(maxsize=CLIENT_OUTPUT_QUEUE_SIZE)

    def start(self) -> None:
        self.sender_task = asyncio.create_task(self._sender_loop())

    def register_job(self, job_id: str, stop: Event) -> bool:
        if not self.active:
            stop.set()
            return False
        self.active_jobs[job_id] = stop
        return True

    def unregister_job(self, job_id: str, stop: Event) -> None:
        if self.active_jobs.get(job_id) is stop:
            self.active_jobs.pop(job_id, None)

    def enqueue_json(self, message: dict[str, object]) -> bool:
        return self._enqueue(OutboundMessage(metadata=message))

    def enqueue_audio_chunk(self, job_id: str, sequence: int, pcm: bytes) -> bool:
        return self._enqueue(
            OutboundMessage(
                metadata={
                    "type": "audio_chunk",
                    "id": job_id,
                    "sequence": sequence,
                    "samples": len(pcm) // 4,
                },
                audio=pcm,
            )
        )

    def _enqueue(self, message: OutboundMessage) -> bool:
        if not self.active:
            return False
        try:
            self.output_queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            self.request_close("client output queue is full")
            return False

    async def _sender_loop(self) -> None:
        while True:
            message = await self.output_queue.get()
            try:
                async with asyncio.timeout(CLIENT_SEND_TIMEOUT):
                    await self.websocket.send_json(message.metadata)
                    if message.audio is not None:
                        await self.websocket.send_bytes(message.audio)
            except Exception as error:
                logger.warning("Closing slow TTS client: %s", error)
                self.request_close("client send failed or timed out")
                return
            finally:
                self.output_queue.task_done()

    def request_close(self, reason: str) -> None:
        if self.closed or self.cleanup_task is not None:
            return
        self.active = False
        for stop in tuple(self.active_jobs.values()):
            stop.set()
        for job_id, text_stream in tuple(self.input_streams.items()):
            if text_stream.scheduled:
                continue
            if (
                self.active_job_ids_registry is not None
                and self.active_job_ids_registry.get(job_id) is text_stream
            ):
                self.active_job_ids_registry.pop(job_id, None)
            if self.input_streams.get(job_id) is text_stream:
                self.input_streams.pop(job_id, None)
            self.unregister_job(job_id, text_stream.stop)
        logger.info("Cleaning up TTS client: %s", reason)
        self.cleanup_task = asyncio.create_task(self._cleanup())

    async def _cleanup(self) -> None:
        if self.sender_task is not None and not self.sender_task.done():
            self.sender_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.sender_task

        while True:
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self.output_queue.task_done()

        with suppress(Exception):
            async with asyncio.timeout(CLIENT_SEND_TIMEOUT):
                await self.websocket.close(code=1011)

        self.sender_task = None
        self.closed = True
        self.cleanup_task = None

    async def close(self, reason: str) -> None:
        self.request_close(reason)
        cleanup_task = self.cleanup_task
        if cleanup_task is not None:
            await cleanup_task


@dataclass
class SynthesisJob:
    id: str
    request: SpeechRequest | None
    client: ClientConnection | None = None
    result: asyncio.Future[bytes] | None = None
    stop: Event = field(default_factory=Event)
    text_stream: TextInputStream | None = None
    owner: object | None = None


def next_stream_chunk(stream: Iterator[bytes]) -> tuple[bytes | None, bool]:
    try:
        return next(stream), False
    except StopIteration:
        return None, True


async def await_stream_chunk(
    stream: Iterator[bytes],
    stop: Event,
) -> tuple[bytes | None, bool]:
    in_flight = asyncio.create_task(
        asyncio.to_thread(next_stream_chunk, stream)
    )
    try:
        return await asyncio.shield(in_flight)
    except asyncio.CancelledError:
        stop.set()
        # The thread cannot be cancelled. Wait until it releases the generator;
        # repeated cancellations are absorbed and its chunk is discarded.
        while not in_flight.done():
            try:
                await asyncio.shield(in_flight)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        with suppress(Exception):
            in_flight.result()
        raise


async def stream_job_to_client(job: SynthesisJob) -> None:
    client = job.client
    if client is None or job.request is None:
        return

    if not client.enqueue_json(
        {
            "type": "audio_start",
            "id": job.id,
            "sample_rate": get_sample_rate(),
            "channels": 1,
            "format": "f32le",
        }
    ):
        return

    stream = stream_audio_chunks(job.request.text, job.request.voice, job.stop)
    sequence = 0
    total_samples = 0
    try:
        while client.active and not job.stop.is_set():
            chunk, complete = await await_stream_chunk(stream, job.stop)
            if complete:
                if client.active and not job.stop.is_set():
                    client.enqueue_json(
                        {
                            "type": "audio_complete",
                            "id": job.id,
                            "chunks": sequence,
                            "samples": total_samples,
                        }
                    )
                return
            if chunk is None:
                continue
            if job.stop.is_set():
                return
            if not client.enqueue_audio_chunk(job.id, sequence, chunk):
                return
            sequence += 1
            total_samples += len(chunk) // 4
    finally:
        job.stop.set()
        close = getattr(stream, "close", None)
        if close is not None:
            # No next() call is still in flight here. Generator close is
            # synchronous, idempotent, and cannot be interrupted by cancellation.
            with suppress(ValueError):
                close()


async def stream_segment_to_client(
    text_stream: TextInputStream,
    client: ClientConnection,
    text: str,
) -> None:
    logger.info(
        "Generating segment %s (%s chars): %s",
        text_stream.id,
        len(text),
        text[:80],
    )
    if not text_stream.audio_started:
        if not client.enqueue_json(
            {
                "type": "audio_start",
                "id": text_stream.id,
                "sample_rate": get_sample_rate(),
                "channels": 1,
                "format": "f32le",
            }
        ):
            return
        text_stream.audio_started = True

    stream = stream_audio_chunks(text, text_stream.voice, text_stream.stop)
    try:
        while client.active and not text_stream.stop.is_set():
            chunk, complete = await await_stream_chunk(stream, text_stream.stop)
            if complete:
                return
            if chunk is None or text_stream.stop.is_set():
                continue
            if not client.enqueue_audio_chunk(
                text_stream.id,
                text_stream.next_sequence,
                chunk,
            ):
                return
            text_stream.next_sequence += 1
            text_stream.total_samples += len(chunk) // 4
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            with suppress(ValueError):
                close()


def release_active_job(
    active_job_owners: ActiveJobOwners,
    job_id: str,
    owner: object,
) -> None:
    if active_job_owners.get(job_id) is owner:
        active_job_owners.pop(job_id, None)


def finalize_text_stream(
    text_stream: TextInputStream,
    client: ClientConnection,
    active_job_owners: ActiveJobOwners,
    *,
    complete_audio: bool,
) -> None:
    owns_stream = client.input_streams.get(text_stream.id) is text_stream
    if (
        complete_audio
        and owns_stream
        and client.active
        and not text_stream.stop.is_set()
        and text_stream.audio_started
    ):
        client.enqueue_json(
            {
                "type": "audio_complete",
                "id": text_stream.id,
                "chunks": text_stream.next_sequence,
                "samples": text_stream.total_samples,
            }
        )
    cancel_idle_timer(text_stream)
    text_stream.stop.set()
    text_stream.buffer = ""
    text_stream.segments.clear()
    text_stream.in_flight_text = None
    text_stream.scheduled = False
    if client.input_streams.get(text_stream.id) is text_stream:
        client.input_streams.pop(text_stream.id, None)
    client.unregister_job(text_stream.id, text_stream.stop)
    release_active_job(
        active_job_owners,
        text_stream.id,
        text_stream,
    )


async def process_incremental_text_stream(
    text_stream: TextInputStream,
    client: ClientConnection,
    active_job_owners: ActiveJobOwners,
) -> bool:
    if (
        text_stream.segments
        and client.active
        and not text_stream.stop.is_set()
    ):
        segment = text_stream.segments.popleft()
        text_stream.in_flight_text = segment
        try:
            await stream_segment_to_client(text_stream, client, segment)
        finally:
            text_stream.in_flight_text = None

    if not client.active or text_stream.stop.is_set():
        finalize_text_stream(
            text_stream,
            client,
            active_job_owners,
            complete_audio=False,
        )
        return False

    if text_stream.segments:
        # One segment per worker turn so a backlogged stream re-enters FIFO
        # instead of draining every committed sentence while others wait.
        return True

    text_stream.scheduled = False
    if text_stream.finished:
        finalize_text_stream(
            text_stream,
            client,
            active_job_owners,
            complete_audio=True,
        )
    else:
        client.enqueue_json(
            {"type": "text_caught_up", "id": text_stream.id}
        )
        arm_idle_timer(text_stream)
    return False


def requeue_incremental_text_stream(
    queue: asyncio.Queue[SynthesisJob],
    text_stream: TextInputStream,
    client: ClientConnection,
    active_job_owners: ActiveJobOwners,
) -> None:
    job = SynthesisJob(
        id=text_stream.id,
        request=None,
        client=client,
        stop=text_stream.stop,
        text_stream=text_stream,
        owner=text_stream,
    )
    try:
        queue.put_nowait(job)
    except asyncio.QueueFull:
        text_stream.scheduled = False
        if client.active:
            client.enqueue_json(
                {
                    "type": "error",
                    "id": text_stream.id,
                    "code": "overloaded",
                    "message": "TTS queue is overloaded; try again shortly",
                }
            )
        finalize_text_stream(
            text_stream,
            client,
            active_job_owners,
            complete_audio=False,
        )
        return
    client.enqueue_json(
        {
            "type": "queued",
            "id": text_stream.id,
            "position": queue.qsize(),
        }
    )


async def synthesis_worker(
    queue: asyncio.Queue[SynthesisJob],
    active_job_owners: ActiveJobOwners,
) -> None:
    while True:
        job = await queue.get()
        incremental = job.text_stream is not None
        try:
            if job.client is not None and not job.client.active:
                if incremental and job.text_stream is job.owner:
                    finalize_text_stream(
                        job.text_stream,
                        job.client,
                        active_job_owners,
                        complete_audio=False,
                    )
                continue
            if job.client is not None and not job.client.enqueue_json(
                {"type": "processing", "id": job.id}
            ):
                if incremental and job.text_stream is job.owner:
                    finalize_text_stream(
                        job.text_stream,
                        job.client,
                        active_job_owners,
                        complete_audio=False,
                    )
                continue

            if incremental and job.client is not None and job.text_stream is job.owner:
                if await process_incremental_text_stream(
                    job.text_stream,
                    job.client,
                    active_job_owners,
                ):
                    requeue_incremental_text_stream(
                        queue,
                        job.text_stream,
                        job.client,
                        active_job_owners,
                    )
            elif job.client is not None:
                await stream_job_to_client(job)
            else:
                if job.request is None:
                    raise ValueError("HTTP synthesis job is missing its request")
                wav = await asyncio.to_thread(
                    generate_wav, job.request.text, job.request.voice
                )
                if job.result is not None and not job.result.done():
                    job.result.set_result(wav)
        except Exception as error:
            logger.exception("Synthesis failed for job %s", job.id)
            if job.client is not None and job.client.active:
                job.client.enqueue_json(
                    {"type": "error", "id": job.id, "message": str(error)}
                )
            if incremental and job.client is not None and job.text_stream is job.owner:
                finalize_text_stream(
                    job.text_stream,
                    job.client,
                    active_job_owners,
                    complete_audio=False,
                )
            if job.result is not None and not job.result.done():
                job.result.set_exception(error)
        finally:
            if not incremental:
                job.stop.set()
                if job.client is not None:
                    job.client.unregister_job(job.id, job.stop)
                if job.owner is not None:
                    release_active_job(
                        active_job_owners,
                        job.id,
                        job.owner,
                    )
            queue.task_done()


@asynccontextmanager
async def lifespan(application: FastAPI):
    await asyncio.to_thread(get_tts)
    application.state.jobs = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    application.state.active_job_ids = {}
    worker = asyncio.create_task(
        synthesis_worker(
            application.state.jobs,
            application.state.active_job_ids,
        )
    )
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
        application.state.active_job_ids.clear()


app = FastAPI(title="TTS", lifespan=lifespan)

template_path = Path(__file__).parent / "templates" / "index.html"
voice_options = "".join(
    f'<option value="{voice}"{" selected" if voice == "eve" else ""}>'
    f"{voice.replace('_', ' ').title()}</option>"
    for voice in ENGLISH_VOICES
)
def render_home() -> str:
    return (
        template_path.read_text(encoding="utf-8")
        .replace("__VOICE_OPTIONS__", voice_options)
        .replace("__MAX_APPEND__", str(PENDING_TEXT_CHARS))
    )


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return render_home()


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ready",
        "queued": str(app.state.jobs.qsize()),
        "capacity": str(MAX_QUEUE_SIZE),
        "active_jobs": str(len(app.state.active_job_ids)),
    }


@app.post("/synthesize")
async def synthesize(request: SpeechRequest) -> Response:
    result = asyncio.get_running_loop().create_future()
    job_id = str(uuid4())
    stop = Event()
    app.state.active_job_ids[job_id] = stop
    try:
        app.state.jobs.put_nowait(
            SynthesisJob(
                id=job_id,
                request=request,
                result=result,
                stop=stop,
                owner=stop,
            )
        )
    except asyncio.QueueFull as error:
        release_active_job(app.state.active_job_ids, job_id, stop)
        raise HTTPException(
            status_code=503, detail="TTS queue is overloaded"
        ) from error
    wav = await result
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts_output.wav"'},
    )


@app.websocket("/ws")
async def synthesis_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    client = ClientConnection(
        websocket,
        active_job_ids_registry=app.state.active_job_ids,
    )
    client.start()

    def send_error(
        job_id: str | None,
        code: str,
        message: str,
        **extra: object,
    ) -> bool:
        return client.enqueue_json(
            {
                "type": "error",
                "id": job_id,
                "code": code,
                "message": message,
                **extra,
            }
        )

    def release_reservation(
        job_id: str,
        stop: Event,
        owner: object,
    ) -> None:
        if app.state.active_job_ids.get(job_id) is not owner:
            return
        if (
            isinstance(owner, TextInputStream)
            and client.input_streams.get(job_id) is not owner
        ):
            return
        stop.set()
        if isinstance(owner, TextInputStream) and owner.scheduled:
            # The synthesis job still owns this stream instance. Keep the
            # process-wide ID reserved until that job's cleanup finishes.
            return
        if isinstance(owner, TextInputStream):
            finalize_text_stream(
                owner,
                client,
                app.state.active_job_ids,
                complete_audio=False,
            )
            return
        client.unregister_job(job_id, stop)
        release_active_job(app.state.active_job_ids, job_id, owner)

    def reserve(job_id: str, stop: Event, owner: object) -> bool:
        if job_id in app.state.active_job_ids:
            send_error(
                job_id,
                "duplicate_job_id",
                "Job ID is already active",
            )
            return False
        app.state.active_job_ids[job_id] = owner
        if not client.register_job(job_id, stop):
            release_active_job(app.state.active_job_ids, job_id, owner)
            return False
        return True

    def admit(job_id: str, request: SpeechRequest, stop: Event) -> bool:
        job = SynthesisJob(
            id=job_id,
            request=request,
            client=client,
            stop=stop,
            owner=stop,
        )
        try:
            app.state.jobs.put_nowait(job)
        except asyncio.QueueFull:
            release_reservation(job_id, stop, stop)
            return send_error(
                job_id,
                "overloaded",
                "TTS queue is overloaded; try again shortly",
            )
        return client.enqueue_json(
            {
                "type": "queued",
                "id": job_id,
                "position": app.state.jobs.qsize(),
            }
        )

    def schedule_text_stream(text_stream: TextInputStream) -> bool:
        if text_stream.scheduled:
            return True
        if not text_stream.segments:
            if text_stream.finished:
                finalize_text_stream(
                    text_stream,
                    client,
                    app.state.active_job_ids,
                    complete_audio=True,
                )
            return True

        text_stream.scheduled = True
        cancel_idle_timer(text_stream)
        job = SynthesisJob(
            id=text_stream.id,
            request=None,
            client=client,
            stop=text_stream.stop,
            text_stream=text_stream,
            owner=text_stream,
        )
        try:
            app.state.jobs.put_nowait(job)
        except asyncio.QueueFull:
            text_stream.scheduled = False
            release_reservation(
                text_stream.id,
                text_stream.stop,
                text_stream,
            )
            return send_error(
                text_stream.id,
                "overloaded",
                "TTS queue is overloaded; try again shortly",
            )
        return client.enqueue_json(
            {
                "type": "queued",
                "id": text_stream.id,
                "position": app.state.jobs.qsize(),
            }
        )

    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type") if isinstance(payload, dict) else None

            if message_type == "text_start":
                try:
                    start = TextStart.model_validate(payload)
                except (TypeError, ValidationError) as error:
                    if not send_error(None, "invalid_text_start", str(error)):
                        break
                    continue

                if len(client.input_streams) >= MAX_CLIENT_INPUT_STREAMS:
                    if not send_error(
                        start.id,
                        "too_many_text_streams",
                        "Too many open text streams",
                    ):
                        break
                    continue

                stop = Event()
                text_stream = TextInputStream(
                    id=start.id,
                    voice=start.voice,
                    stop=stop,
                )
                if not reserve(start.id, stop, text_stream):
                    if not client.active:
                        break
                    continue
                def expire_idle(stream: TextInputStream = text_stream) -> None:
                    stream.idle_handle = None
                    if (
                        stream.finished
                        or stream.scheduled
                        or stream.stop.is_set()
                        or client.input_streams.get(stream.id) is not stream
                        or not client.active
                    ):
                        return
                    send_error(
                        stream.id,
                        "idle_timeout",
                        f"Text stream idle for {TEXT_IDLE_TIMEOUT:g} seconds",
                    )
                    finalize_text_stream(
                        stream,
                        client,
                        app.state.active_job_ids,
                        complete_audio=False,
                    )

                text_stream.idle_callback = expire_idle
                client.input_streams[start.id] = text_stream
                arm_idle_timer(text_stream)
                if not client.enqueue_json(
                    {"type": "text_started", "id": start.id}
                ):
                    break
                continue

            if message_type == "text_append":
                try:
                    append = TextAppend.model_validate(payload)
                except (TypeError, ValidationError) as error:
                    if not send_error(None, "invalid_text_append", str(error)):
                        break
                    continue

                text_stream = client.input_streams.get(append.id)
                if text_stream is None or text_stream.stop.is_set():
                    if not send_error(
                        append.id,
                        "unknown_text_stream",
                        "Text stream is not active",
                    ):
                        break
                    continue
                if text_stream.characters + len(append.text) > MAX_TEXT_LENGTH:
                    release_reservation(
                        append.id,
                        text_stream.stop,
                        text_stream,
                    )
                    if not send_error(
                        append.id,
                        "text_too_long",
                        f"text exceeds {MAX_TEXT_LENGTH} characters",
                    ):
                        break
                    continue
                pending = text_stream.pending_text_chars()
                if len(append.text) > PENDING_TEXT_CHARS:
                    if not text_stream.scheduled and not text_stream.finished:
                        arm_idle_timer(text_stream)
                    if not send_error(
                        append.id,
                        "append_too_large",
                        "Append exceeds the pending text limit; split this text and send smaller pieces",
                        pending=pending,
                        limit=PENDING_TEXT_CHARS,
                        rejected=len(append.text),
                        max_append=PENDING_TEXT_CHARS,
                    ):
                        break
                    continue
                if pending + len(append.text) > PENDING_TEXT_CHARS:
                    if text_stream.buffer.strip() and not text_stream.segments:
                        if text_stream.flush_buffer() and not schedule_text_stream(
                            text_stream
                        ):
                            break
                        pending = text_stream.pending_text_chars()
                    if not text_stream.scheduled and not text_stream.finished:
                        arm_idle_timer(text_stream)
                    if not send_error(
                        append.id,
                        "pending_text_limit",
                        "Pending text limit reached; retry this append after some text is synthesized",
                        pending=pending,
                        limit=PENDING_TEXT_CHARS,
                        rejected=len(append.text),
                    ):
                        break
                    continue
                try:
                    committed = text_stream.append(append.text)
                except ValueError as error:
                    release_reservation(
                        append.id,
                        text_stream.stop,
                        text_stream,
                    )
                    if not send_error(
                        append.id,
                        "text_too_long",
                        str(error),
                    ):
                        break
                    continue
                if not client.enqueue_json(
                    {
                        "type": "text_appended",
                        "id": append.id,
                        "characters": text_stream.characters,
                        "segments_committed": committed,
                        "pending": text_stream.pending_text_chars(),
                        "limit": PENDING_TEXT_CHARS,
                    }
                ):
                    break
                if committed:
                    if not schedule_text_stream(text_stream):
                        break
                else:
                    arm_idle_timer(text_stream)
                continue

            if message_type == "text_finish":
                try:
                    finish = TextFinish.model_validate(payload)
                except (TypeError, ValidationError) as error:
                    if not send_error(None, "invalid_text_finish", str(error)):
                        break
                    continue

                text_stream = client.input_streams.get(finish.id)
                if text_stream is None or text_stream.stop.is_set():
                    if not send_error(
                        finish.id,
                        "unknown_text_stream",
                        "Text stream is not active",
                    ):
                        break
                    continue
                try:
                    cancel_idle_timer(text_stream)
                    text_stream.finish()
                except ValueError as error:
                    release_reservation(
                        finish.id,
                        text_stream.stop,
                        text_stream,
                    )
                    if not send_error(
                        finish.id,
                        "invalid_text_stream",
                        str(error),
                    ):
                        break
                    continue
                if not schedule_text_stream(text_stream):
                    break
                continue

            # Backwards-compatible one-shot text request.
            try:
                request = SpeechRequest.model_validate(payload)
                job_id = str(payload.get("id") or uuid4())
            except (AttributeError, TypeError, ValidationError) as error:
                if not send_error(None, "invalid_request", str(error)):
                    break
                continue

            stop = Event()
            if not reserve(job_id, stop, stop):
                if not client.active:
                    break
                continue
            if not admit(job_id, request, stop):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await client.close("websocket disconnected")


sdk_dir = Path(__file__).parent / "sdk"
app.mount("/sdk", StaticFiles(directory=sdk_dir), name="sdk")
