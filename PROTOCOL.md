# WebSocket protocol

Browser and agent integrations should use [`sdk/tts-client.js`](sdk/tts-client.js)
(`startStream` / `appendText` / `finishText` / `onAudio` / `onError`) instead of
speaking this framing directly. The SDK transport is swappable so a later gRPC
gateway can carry the same control objects.

## Text input

Each message is JSON. A text stream reserves its job ID at `text_start`.
Complete sentences are admitted immediately, so audio can arrive before
`text_finish`.

```json
{"type":"text_start","id":"job-1","voice":"eve"}
{"type":"text_append","id":"job-1","text":"Hello "}
{"type":"text_append","id":"job-1","text":"world."}
{"type":"text_finish","id":"job-1"}
```

The server commits a sentence at `.`, `?`, or `!` only when that sentence
fits in `TTS_TEXT_SEGMENT_CHARS` (80 by default). A longer sentence is split
at the last whitespace within the limit, or hard-split when no whitespace
exists, so one generation call cannot exceed the cap. Any remaining text is
committed by `text_finish`. Decimal points are treated the same as sentence
periods; avoiding a split in `3.14` would require lookahead into later
appends.

All committed segments share one `audio_start`/`audio_complete` pair and one
continuous `audio_chunk.sequence`. The worker synthesizes one committed
segment per turn and re-enters the global FIFO if more segments remain, so a
backlogged stream cannot monopolize synthesis. When a stream catches up with
its input, it also releases the worker and re-enters the FIFO after more text
is committed.

A stream may accumulate up to `TTS_MAX_TEXT_LENGTH` characters (50000 by
default). Unfinished work is also capped per stream: accepted text that is
still buffered, queued, or being synthesized cannot exceed
`TTS_PENDING_TEXT_CHARS` (400 by default, five segment caps). An append
that would pass the remaining budget is rejected with a non-terminal
`pending_text_limit` error:

```json
{"type":"error","id":"job-1","code":"pending_text_limit","pending":400,"limit":400,"rejected":48}
```

Retry the same append after some pending text has been synthesized, and do
not send later text until it is accepted. If that append would fit the
pending cap but not the remaining budget, and the stream has uncommitted
buffer with no queued segments, the server commits that buffer and
schedules synthesis before returning `pending_text_limit`, so capacity can
free. A single append larger than
`TTS_PENDING_TEXT_CHARS` can never be accepted; the server rejects it with
`append_too_large` and the client must split that text and send smaller
pieces:

```json
{"type":"error","id":"job-1","code":"append_too_large","pending":0,"limit":400,"rejected":5000,"max_append":400}
```

`text_appended` includes the current `pending` count and `limit` so a
client can pace itself. `pending_text_limit` does not yet include
`retry_after_ms`.

After `text_start`, or after the worker catches up and is waiting for more
text, the server closes the stream with `idle_timeout` if no `text_append`
or `text_finish` arrives within `TTS_TEXT_IDLE_TIMEOUT` seconds (20 by
default).

The legacy one-shot form remains supported:

```json
{"id":"job-1","voice":"eve","text":"Hello world."}
```

## Audio output

The server emits JSON metadata and binary mono float32 little-endian PCM:

```json
{"type":"audio_start","id":"job-1","sample_rate":24000,"channels":1,"format":"f32le"}
{"type":"audio_chunk","id":"job-1","sequence":0,"samples":1920}
```

The binary payload immediately following `audio_chunk` contains that chunk.
The stream ends with:

```json
{"type":"audio_complete","id":"job-1","chunks":10,"samples":19200}
```
