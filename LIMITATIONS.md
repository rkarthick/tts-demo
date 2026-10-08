# Current limitations

- Job IDs are unique process-wide while active. Two different clients cannot use
  the same active ID, even though their WebSocket sessions are independent.
- An incremental text stream is closed with `idle_timeout` if the client sends
  no further `text_append` or `text_finish` within `TTS_TEXT_IDLE_TIMEOUT`
  after start or after the worker catches up.
- `pending_text_limit` is a non-terminal backpressure signal. The server does
  not yet return `retry_after_ms`; a later heuristic could tell the client
  how long to wait before retrying the rejected append. The JavaScript SDK
  retries that code with backoff (capped at 5s) and does not bound the number
  of attempts, so `appendText` can wait indefinitely while the server keeps
  rejecting. Future work: cap retries and fail the stream after N attempts.
- An append larger than `TTS_PENDING_TEXT_CHARS` is rejected with
  `append_too_large`. The stream stays open; the client must split that
  packet and send pieces no larger than `max_append`.
- Pocket TTS warns above 50 tokens and can stall or skip words on larger
  chunks. Dense academic English hit 51 tokens at 120 characters, so the
  segment cap is 80 characters. Each generate call still holds the process
  lock, so a long document can feel stalled even when chunks are in budget.
- If a sentence has no `.`, `?`, or `!` in a pending-text block
  (`TTS_PENDING_TEXT_CHARS`, 400 by default), a single append of that block
  is rejected as
  `append_too_large` and is never accepted unless the client splits it. Even
  after a split, remainder without punctuation is not a completed sentence
  until `text_finish` or a later size boundary.
- Queue admission is global and has no per-client quota. One connection can
  fill the entire waiting queue and temporarily prevent other clients from
  submitting work. Once admitted, incremental streams yield the worker after
  each segment so a backlog cannot monopolize synthesis.
- Each committed incremental text segment is a separate Pocket TTS model call.
  The server presents their PCM chunks as one ordered audio stream, but a small
  audible discontinuity between segments is possible.
- Sentence detection uses `.`, `?`, and `!` followed by whitespace or the end
  of the current append. It does not perform language-aware abbreviation
  detection.
- Decimal numbers can be split as sentence boundaries. A value such as `3.14`
  looks like a completed sentence once `.` arrives, and later digits may be
  committed separately. Distinguishing `3.14` from `3.` requires lookahead into
  the next append, which the current commit rules do not do.
