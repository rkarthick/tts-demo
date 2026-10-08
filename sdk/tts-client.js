/**
 * TTS streaming client.
 *
 * Caller API (stable; keep this if the wire becomes gRPC):
 *   const client = new TtsClient({ url });
 *   await client.connect();
 *   const stream = await client.startStream({ voice, onAudio, onError });
 *   await stream.appendText("Hello world.");
 *   await stream.finishText();
 *
 * Transport is the only WebSocket-specific piece. A gateway can implement
 * the same contract over a bidirectional gRPC stream: send control objects
 * (`text_start` / `text_append` / `text_finish`) and deliver control events
 * plus binary PCM payloads. Do not leak framing into the page or agent.
 */

const RETRY_BASE_MS = 200;
const RETRY_MAX_MS = 5000;
const APPEND_ACK_TIMEOUT_MS = 5000;

export class TtsError extends Error {
  constructor(message, fields = {}) {
    super(message);
    this.name = "TtsError";
    this.code = fields.code ?? "error";
    this.id = fields.id;
    this.pending = fields.pending;
    this.limit = fields.limit;
    this.rejected = fields.rejected;
    this.retryAfterMs = fields.retry_after_ms ?? fields.retryAfterMs;
    this.maxAppend = fields.max_append ?? fields.maxAppend ?? fields.limit;
    this.pieces = fields.pieces;
    this.retryDelayMs = fields.retryDelayMs;
    this.recoverable = fields.recoverable ?? (
      this.code === "pending_text_limit"
      || this.code === "append_too_large"
    );
  }
}

export function defaultWebSocketUrl(base) {
  if (base) {
    const url = new URL("/ws", base);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
  }
  if (typeof location === "undefined") {
    return "ws://127.0.0.1:8000/ws";
  }
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/ws`;
}

export class WebSocketTransport {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this._pendingChunk = null;
    this.onControl = null;
    this.onBinary = null;
    this.onClose = null;
    this.onTransportError = null;
  }

  get ready() {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  connect() {
    if (this.ready) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(this.url);
      socket.binaryType = "arraybuffer";
      const onOpen = () => {
        socket.removeEventListener("open", onOpen);
        socket.removeEventListener("error", onConnectError);
        resolve();
      };
      const onConnectError = () => {
        socket.removeEventListener("open", onOpen);
        socket.removeEventListener("error", onConnectError);
        reject(new TtsError("WebSocket failed to connect", { code: "connect_failed" }));
      };
      socket.addEventListener("open", onOpen);
      socket.addEventListener("error", onConnectError);
      socket.addEventListener("message", (event) => {
        this._onMessage(event).catch((error) => {
          this.onTransportError?.(error);
        });
      });
      socket.addEventListener("close", () => {
        if (this.socket === socket) this.socket = null;
        this._pendingChunk = null;
        this.onClose?.();
      });
      this.socket = socket;
    });
  }

  send(message) {
    if (!this.ready) {
      throw new TtsError("Transport is not connected", { code: "disconnected" });
    }
    this.socket.send(JSON.stringify(message));
  }

  close() {
    this.socket?.close();
  }

  async _onMessage(event) {
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "audio_chunk") this._pendingChunk = message;
      this.onControl?.(message);
      return;
    }
    const metadata = this._pendingChunk;
    this._pendingChunk = null;
    if (!metadata) {
      throw new TtsError("Received audio without chunk metadata", {
        code: "protocol",
      });
    }
    const buffer = event.data instanceof ArrayBuffer
      ? event.data
      : await event.data.arrayBuffer();
    this.onBinary?.(metadata, buffer);
  }
}

function decodeF32le(buffer) {
  const view = new DataView(buffer);
  const samples = new Float32Array(buffer.byteLength / 4);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getFloat32(index * 4, true);
  }
  return samples;
}

function retryDelayMs(error, attempt) {
  const retryAfter = Number(error.retryAfterMs);
  if (Number.isFinite(retryAfter) && retryAfter > 0) return retryAfter;
  return Math.min(RETRY_MAX_MS, RETRY_BASE_MS * (2 ** attempt));
}

export function splitPacket(packet, maxSize) {
  const limit = Math.max(
    1,
    Math.floor(Number(maxSize)) || Math.max(1, Math.floor(packet.length / 2)),
  );
  if (packet.length <= limit) {
    if (packet.length <= 1) return [packet];
    const mid = Math.ceil(packet.length / 2);
    return [packet.slice(0, mid), packet.slice(mid)];
  }
  const pieces = [];
  let offset = 0;
  while (offset < packet.length) {
    let end = Math.min(packet.length, offset + limit);
    if (end < packet.length) {
      const window = packet.slice(offset, end);
      const breakAt = window.search(/\s+\S*$/);
      if (breakAt >= 1) end = offset + breakAt + 1;
    }
    if (end <= offset) end = Math.min(packet.length, offset + limit);
    pieces.push(packet.slice(offset, end));
    offset = end;
  }
  return pieces;
}

function emit(handler, payload) {
  if (!handler) return;
  try {
    handler(payload);
  } catch {
    // Caller callbacks must not break the stream.
  }
}

export class TtsClient {
  constructor({
    url,
    transport,
    onClose,
    appendAckTimeoutMs = APPEND_ACK_TIMEOUT_MS,
  } = {}) {
    this.appendAckTimeoutMs = Math.max(1, Number(appendAckTimeoutMs) || APPEND_ACK_TIMEOUT_MS);
    this.transport = transport ?? new WebSocketTransport(url ?? defaultWebSocketUrl());
    this._streams = new Map();
    this._onClose = onClose;
    this.transport.onControl = (message) => this._dispatchControl(message);
    this.transport.onBinary = (metadata, buffer) => this._dispatchBinary(metadata, buffer);
    this.transport.onClose = () => this._handleClose();
    this.transport.onTransportError = (error) => {
      const wrapped = error instanceof TtsError
        ? error
        : new TtsError(error.message, { code: "protocol" });
      for (const stream of [...this._streams.values()]) stream._fail(wrapped);
    };
  }

  get connected() {
    return this.transport.ready;
  }

  connect() {
    return this.transport.connect();
  }

  close() {
    this.transport.close();
  }

  async startStream({
    id,
    voice = "eve",
    onAudio,
    onError,
    onEvent,
    appendAckTimeoutMs,
  } = {}) {
    if (!this.connected) await this.connect();
    const stream = new TtsStream(this, {
      id: id ?? crypto.randomUUID(),
      voice,
      onAudio,
      onError,
      onEvent,
      appendAckTimeoutMs: appendAckTimeoutMs ?? this.appendAckTimeoutMs,
    });
    this._streams.set(stream.id, stream);
    try {
      await stream._start();
    } catch (error) {
      this._streams.delete(stream.id);
      throw error;
    }
    return stream;
  }

  _dispatchControl(message) {
    if (!message?.id) return;
    this._streams.get(message.id)?._handleControl(message);
  }

  _dispatchBinary(metadata, buffer) {
    this._streams.get(metadata.id)?._handleAudio(metadata, buffer);
  }

  _release(id) {
    this._streams.delete(id);
  }

  _handleClose() {
    const error = new TtsError("Transport disconnected", { code: "disconnected" });
    for (const stream of this._streams.values()) stream._fail(error);
    this._streams.clear();
    emit(this._onClose, error);
  }
}

export class TtsStream {
  constructor(client, {
    id,
    voice,
    onAudio,
    onError,
    onEvent,
    appendAckTimeoutMs = APPEND_ACK_TIMEOUT_MS,
  }) {
    this.id = id;
    this.voice = voice;
    this.client = client;
    this.onAudio = onAudio;
    this.onError = onError;
    this.onEvent = onEvent;
    this.appendAckTimeoutMs = Math.max(1, Number(appendAckTimeoutMs) || APPEND_ACK_TIMEOUT_MS);
    this.pending = 0;
    this.limit = null;
    this._waiters = [];
    this._writeChain = Promise.resolve();
    this._closed = null;
    this._outcomeSettled = false;
    this._outcome = new Promise((resolve, reject) => {
      this._resolveOutcome = resolve;
      this._rejectOutcome = reject;
    });
    this._outcome.catch(() => undefined);
    this._sampleRate = null;
    this._channels = 1;
    this._format = "f32le";
  }

  get closed() {
    return this._closed;
  }

  appendText(text) {
    if (this._closed) return Promise.reject(this._closed);
    return this._enqueueWrite(() => this._appendNow(text));
  }

  finishText() {
    return this._enqueueWrite(() => this._finishNow());
  }

  _enqueueWrite(work) {
    const run = this._writeChain.then(() => work());
    this._writeChain = run.then(() => undefined, () => undefined);
    return run;
  }

  async _appendNow(text) {
    if (this._closed) throw this._closed;
    if (text == null || text === "") return;
    const queue = [String(text)];
    let attempt = 0;
    while (queue.length) {
      if (this._closed) throw this._closed;
      const current = queue[0];
      const appended = this._wait("text_appended", this.appendAckTimeoutMs);
      this.client.transport.send({
        type: "text_append",
        id: this.id,
        text: current,
      });
      try {
        await appended;
        queue.shift();
        attempt = 0;
      } catch (error) {
        if (error.code === "append_timeout") {
          this._fail(error);
          this.client.close();
          throw error;
        }
        if (error.code === "append_too_large") {
          const pieces = splitPacket(current, error.maxAppend);
          if (pieces.length === 1 && pieces[0] === current) throw error;
          error.pieces = pieces.map((part) => part.length);
          emit(this.onError, error);
          queue.splice(0, 1, ...pieces);
          attempt = 0;
          continue;
        }
        if (error.code !== "pending_text_limit") throw error;
        const delay = retryDelayMs(error, attempt);
        attempt += 1;
        error.retryDelayMs = delay;
        emit(this.onError, error);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  _finishNow() {
    if (this._closed && this._closed.code !== "finished" && this._closed.code !== "complete") {
      throw this._closed;
    }
    if (this._closed?.code !== "finished" && this._closed?.code !== "complete") {
      this.client.transport.send({ type: "text_finish", id: this.id });
      this._closed = new TtsError("text stream is already finished", {
        code: "finished",
        id: this.id,
      });
    }
    return this._outcome;
  }

  _succeed() {
    if (this._outcomeSettled) return;
    this._outcomeSettled = true;
    this._resolveOutcome();
  }

  _rejectOutcomeWith(error) {
    if (this._outcomeSettled) return;
    this._outcomeSettled = true;
    this._rejectOutcome(error);
  }

  _start() {
    const started = this._wait("text_started");
    this.client.transport.send({
      type: "text_start",
      id: this.id,
      voice: this.voice,
    });
    return started;
  }

  _wait(type, timeoutMs) {
    return new Promise((resolve, reject) => {
      const waiter = { type };
      let timer = null;
      waiter.resolve = (value) => {
        if (timer !== null) clearTimeout(timer);
        resolve(value);
      };
      waiter.reject = (error) => {
        if (timer !== null) clearTimeout(timer);
        reject(error);
      };
      if (timeoutMs != null) {
        timer = setTimeout(() => {
          const index = this._waiters.indexOf(waiter);
          if (index >= 0) this._waiters.splice(index, 1);
          waiter.reject(new TtsError("append was not acknowledged", {
            code: "append_timeout",
            id: this.id,
          }));
        }, timeoutMs);
      }
      this._waiters.push(waiter);
    });
  }

  _signalError(error) {
    emit(this.onError, error);
  }

  _fail(error) {
    const cleanClose = (
      !this._closed
      || this._closed.code === "finished"
      || this._closed.code === "complete"
    );
    const duplicate = !cleanClose && this._closed?.code === error.code;
    if (cleanClose) this._closed = error;
    for (const waiter of this._waiters.splice(0)) waiter.reject(error);
    if (!duplicate) this._signalError(error);
    this._rejectOutcomeWith(error);
    this.client._release(this.id);
  }

  _resolveWaiters(message) {
    for (let index = this._waiters.length - 1; index >= 0; index -= 1) {
      const waiter = this._waiters[index];
      if (message.type === "error") {
        if (
          (message.code === "pending_text_limit" || message.code === "append_too_large")
          && waiter.type !== "text_appended"
        ) {
          continue;
        }
        this._waiters.splice(index, 1);
        waiter.reject(new TtsError(message.message || message.code, message));
      } else if (waiter.type === message.type) {
        this._waiters.splice(index, 1);
        waiter.resolve(message);
      }
    }
  }

  _handleControl(message) {
    this._resolveWaiters(message);

    if (message.type === "text_appended") {
      this.pending = message.pending;
      this.limit = message.limit;
      emit(this.onEvent, message);
      return;
    }

    if (message.type === "audio_start") {
      this._sampleRate = message.sample_rate;
      this._channels = message.channels ?? 1;
      this._format = message.format ?? "f32le";
      emit(this.onAudio, {
        type: "start",
        id: this.id,
        sampleRate: this._sampleRate,
        channels: this._channels,
        format: this._format,
      });
      return;
    }

    if (message.type === "audio_complete") {
      emit(this.onAudio, {
        type: "complete",
        id: this.id,
        chunks: message.chunks,
        samples: message.samples,
      });
      if (!this._closed || this._closed.code === "finished") {
        this._closed = new TtsError("audio stream is complete", {
          code: "complete",
          id: this.id,
        });
      }
      for (const waiter of this._waiters.splice(0)) waiter.reject(this._closed);
      this._succeed();
      this.client._release(this.id);
      return;
    }

    if (message.type === "error") {
      const error = new TtsError(message.message || message.code, message);
      if (!error.recoverable) this._fail(error);
      return;
    }

    emit(this.onEvent, message);
  }

  _handleAudio(metadata, buffer) {
    emit(this.onAudio, {
      type: "chunk",
      id: this.id,
      sequence: metadata.sequence,
      samples: metadata.samples,
      sampleRate: this._sampleRate,
      channels: this._channels,
      format: this._format,
      pcm: decodeF32le(buffer),
    });
  }
}

const publicApi = {
  TtsClient,
  TtsStream,
  TtsError,
  WebSocketTransport,
  defaultWebSocketUrl,
  splitPacket,
};

if (typeof window !== "undefined") {
  Object.assign(window, publicApi);
}
