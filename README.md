# TTS demo

A local text-to-speech server built on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts). It serves a browser page, a one-shot WAV endpoint, and a WebSocket stream that speaks text as it arrives. One worker synthesizes audio inside the process, so concurrent clients share a single generation queue.

## Run it locally

Python 3.12 or newer. The first start downloads the Pocket TTS weights into `./model`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Pick a voice, type or paste text, and play the stream. `GET /health` returns `ready` once the model is loaded.

Write a WAV from the command line:

```bash
.venv/bin/python tts.py "Hello from Pocket TTS."
```

That writes `tts_output.wav` in the current directory. Pass a path to a UTF-8 text file instead of a quoted string to read the script from disk.

## Docker

The image uses Python 3.12 and the CPU PyTorch build. The image build downloads the model and warms every built-in voice, so the first build takes a while.

```bash
docker compose up --build
```

The server listens on http://127.0.0.1:8000. Model files stay inside the container at `/models`.

## HTTP

`POST /synthesize` accepts JSON and returns a WAV file. The default voice is `eve`.

```bash
curl -X POST http://127.0.0.1:8000/synthesize \
  -H 'content-type: application/json' \
  -d '{"text":"Hello world.","voice":"eve"}' \
  --output hello.wav
```

A full queue responds with HTTP 503.

## WebSocket

Connect to `ws://127.0.0.1:8000/ws`. Incremental clients send `text_start`, `text_append`, and `text_finish`. The server answers with JSON metadata and binary float32 PCM. Message shapes, backpressure, and idle timeout are in [PROTOCOL.md](PROTOCOL.md).

Browser and agent code should use the JavaScript client instead of the framing directly:

```js
import { TtsClient } from "/sdk/tts-client.js";
```

The client API is in [sdk/README.md](sdk/README.md). The page at `/` is one consumer of that file.

A one-shot WebSocket message is still accepted:

```json
{"id":"job-1","voice":"eve","text":"Hello world."}
```

## Voices

Built-in English voices: alba, anna, azelma, bill_boerst, caro_davy, charles, cosette, eponine, eve, fantine, george, jane, javert, jean, marius, mary, michael, paul, peter_yearsley, stuart_bell, vera.

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `POCKET_TTS_MODEL_DIR` | `./model` | Where Pocket TTS and Hugging Face cache weights |
| `TTS_MAX_QUEUE_SIZE` | `10` | Waiting jobs before new work is rejected |
| `TTS_CLIENT_OUTPUT_QUEUE_SIZE` | `10` | Outbound audio chunks buffered per connection |
| `TTS_CLIENT_SEND_TIMEOUT` | `5` | Seconds before a slow client send fails |
| `TTS_CLIENT_INPUT_STREAMS` | `10` | Open text streams allowed on one connection |
| `TTS_TEXT_SEGMENT_CHARS` | `80` | Maximum characters in one generation call |
| `TTS_PENDING_TEXT_CHARS` | `400` | Unfinished text a stream may hold |
| `TTS_MAX_TEXT_LENGTH` | `50000` | Maximum characters in one stream |
| `TTS_TEXT_IDLE_TIMEOUT` | `20` | Seconds of silence before an idle stream closes |

Known gaps in queue fairness, sentence splitting, and playback are listed in [LIMITATIONS.md](LIMITATIONS.md).

## Tests and the harness

Install the test extras, then run the suite. Most tests stub the model. Set `RUN_REAL_TTS_TEST=1` to include the live inference smoke test.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/pytest
RUN_REAL_TTS_TEST=1 .venv/bin/pytest -m smoke
```

With a server already running, the harness drives concurrent WebSocket clients and prints playback timing:

```bash
.venv/bin/python harness.py --url http://127.0.0.1:8000 --clients 4 --burst-extra 4 --repeats 3
```

Saved runs and how to read them are in [PERFORMANCE.md](PERFORMANCE.md).
