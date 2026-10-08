# TTS JavaScript SDK

Browser library for incremental speech streams. The demo page is a consumer of
this file, not a second copy of the protocol.

```js
import { TtsClient } from "/sdk/tts-client.js";

const client = new TtsClient();
await client.connect();

const stream = await client.startStream({
  voice: "eve",
  onAudio(event) {
    if (event.type === "chunk") play(event.pcm, event.sampleRate);
  },
  onError(error) {
    if (error.recoverable) return; // pending limit / split; appendText retries
    console.error(error);
  },
});

await stream.appendText("Hello from an agent.");
await stream.finishText();
```

## Surface

| Method | Role |
| --- | --- |
| `startStream({ voice, onAudio, onError })` | Reserve a job ID and start text input |
| `appendText(text)` | Send more text; waits until the server accepts it |
| `finishText()` | End text input, then settle on `audio_complete` or a later error |
| `onAudio` | `start` / `chunk` (`pcm` is `Float32Array`) / `complete` |
| `onError` | Fatal errors, plus recoverable backpressure while retrying |

`appendText` is serialized per stream: overlapping calls wait their turn,
and each waits until the server accepts that text (retrying
`pending_text_limit` and splitting `append_too_large`). If `text_appended`
does not arrive within `appendAckTimeoutMs` (5s by default), `appendText`
rejects with `append_timeout` and the client disconnects. `finishText`
waits for earlier appends, sends `text_finish`, then settles when audio
completes. A synthesis, protocol, or disconnect error after finish rejects
that promise and is also delivered to `onError`. Later `appendText` calls
still reject (`finished`, the error code, `complete`, or `disconnected`).
Wire details are in [`../PROTOCOL.md`](../PROTOCOL.md).

## Later: gRPC gateway

Keep this API. Replace `WebSocketTransport` with a bidirectional gRPC client
that maps the same control objects (`text_start`, `text_append`, `text_finish`)
and the same events (`audio_start`, `audio_chunk` + payload, `audio_complete`,
`error`) onto proto messages. The page, agents, and retry rules stay the same;
only the transport changes.
