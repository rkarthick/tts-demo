# Performance

These are four saved harness runs on a Mac mini M4 with 16 GB of memory, using either native macOS Python or Docker on the same machine. They show where playback begins to degrade for a single synthesis worker. They are development measurements, not a controlled benchmark or a production capacity guarantee.

## Environment and workload

Both runtimes use Pocket TTS with CPU inference and serialize model generation within one server process. The native environment uses the macOS PyTorch build with Apple Accelerate; Docker uses a Linux CPU build inside Docker Desktop. MPS support in the native PyTorch build does not mean the application uses the GPU.

The recorded diagnostics were Python 3.14.6 / PyTorch 2.14.1 on macOS and Python 3.12.15 / PyTorch 2.14.1+cpu in Docker. Both reported one PyTorch intra-op thread. Docker's CPU and memory allocation was not captured with these runs.

The table uses only the `multiple_concurrent_text` scenario from each report. All four reports record a waiting-queue capacity of 10. Native runs repeated the harness text three times per client; the Docker run repeated it twice. Here, "repeats" means more text in the same stream, not independent benchmark trials.

## Observed results

Time to first audio is measured at the harness client. Stall duration is summed across all clients, so it can exceed the scenario's wall-clock duration; a stall means simulated playback ran out of received audio after starting.

| Runtime / raw report | Clients completed | Text repeats | First audio: mean / max (ms) | Playback stalls | Total stall time (s) | Longest stall (s) | Scenario time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Native Mac, 4 clients](src/harness_4clients_4bursts_3repeats.out) | 4 / 4 | 3 | 821.1 / 1617.4 | 0 | 0 | 0 | 19.68 |
| [Native Mac, 6 clients](src/harness_4clients_6bursts_3repeats.out) | 6 / 6 | 3 | 2370.3 / 4313.8 | 2 | 0.978 | 0.534 | 33.72 |
| [Native Mac, 8 clients](src/harness_8clients_4bursts_3repeats.out) | 8 / 8 | 3 | 2300.4 / 4330.0 | 19 | 106.238 | 13.600 | 54.72 |
| [Docker, 2 clients](src/docker_harness_2clients_4bursts_3repeats.out) | 2 / 2 | 2 | 1493.3 / 2860.4 | 10 | 13.749 | 1.742 | 36.40 |

The report contents, not filenames, determine the settings above. The native file named `harness_4clients_6bursts_3repeats.out` actually records 6 clients and 4 extra burst clients. The Docker filename says `3repeats`, but its report records 2.

## What the results show

- **Native Mac:** Four concurrent streams completed without simulated playback stalls in the saved run. Six clients introduced short stalls; eight produced substantial playback interruptions. Four is therefore the highest stall-free concurrency among these saved concurrent runs, not a proven safe admission limit. Even at four clients, the slowest first audio took about 1.6 seconds.
- **Docker:** Two concurrent streams completed, but playback stalled ten times despite the shorter text workload. This run shows that native Mac capacity cannot be used as the container's capacity estimate. Different runtime builds and Docker resource allocation are possible contributors; these measurements do not isolate their individual effects.
- **Completion versus playback:** All four concurrent scenarios were marked `PASS`, with no overload rejections or recorded failures. That means their functional checks passed, not that they met a real-time playback target.
- **Overload and cleanup:** In each saved report, the separate burst scenario recorded three overload rejections and completed a follow-up stream after the queue drained. The disconnect scenario also completed its surviving stream. These checks exercise rejection and recovery behavior, but do not establish sustained-load reliability.

## Limits of this comparison

The native and Docker rows use different concurrency and text lengths, so they should not be used to claim an exact throughput speedup. Each row is one saved scenario run, with no confidence interval, repeated-trial validation, or complete record of the code revision, server settings, warm-up state, and background machine load.

The playback model measures delivery underruns, not audible quality. Pauses encoded by the TTS model or discontinuities between separately generated segments can still be audible when the harness reports zero stalls.

## Reproducing the workloads

Start one server using the [README instructions](README.md), warm the selected voice with a short request, and run the harness from `src/`. These commands reproduce the recorded workload arguments; they cannot reconstruct unrecorded environment settings or guarantee the same measurements.

```bash
# Native Mac workloads, with the native server running
.venv/bin/python harness.py --url http://127.0.0.1:8000 --clients 4 --burst-extra 4 --repeats 3
.venv/bin/python harness.py --url http://127.0.0.1:8000 --clients 6 --burst-extra 4 --repeats 3
.venv/bin/python harness.py --url http://127.0.0.1:8000 --clients 8 --burst-extra 4 --repeats 3

# Docker workload, with only the Docker server running
# Use .venv/bin/python instead if the lightweight harness environment is absent.
.harness-venv/bin/python harness.py --url http://127.0.0.1:8000 --clients 2 --burst-extra 4 --repeats 2
```
