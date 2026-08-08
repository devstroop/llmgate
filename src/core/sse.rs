//! Minimal SSE framing parser (protocol-agnostic).
//!
//! Consumes raw byte chunks of an SSE stream and yields complete event
//! payloads: the concatenation of all `data:` lines of an event, terminated
//! by a blank line. Comment lines (`:`), `event:`, `id:`, `retry:` fields are
//! ignored; `[DONE]` and other payloads pass through verbatim.
//!
//! A cap bounds how many bytes can be buffered between line terminators so a
//! broken or malicious upstream cannot exhaust memory with an endless
//! unterminated line.

/// Maximum bytes buffered before a newline terminator. If exceeded, the
/// buffer is discarded to bound memory. Well-formed SSE events are many
/// orders of magnitude smaller; this only fires on a broken/malicious peer.
const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub struct SseFraming {
    buffer: Vec<u8>,
    /// Accumulated `data:` bytes of the current event (a single buffer:
    /// per-line String allocations would let a flood of tiny lines defeat
    /// the byte cap with allocator overhead).
    data_lines: Vec<u8>,
    /// Cumulative bytes of `data_lines` for the current event, so the cap
    /// also bounds events built from many newline-terminated lines (the
    /// unterminated-line buffer alone would not).
    data_lines_bytes: usize,
    /// Count of dropped non-UTF-8 data lines, for rate-limited logging.
    invalid_utf8_lines: u64,
}

impl SseFraming {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns the complete event payloads found (in order).
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        // Reject an oversized chunk BEFORE copying it, so the cap actually
        // bounds peak memory (a single huge chunk must not be buffered just
        // to be discarded).
        if chunk.len() > MAX_BUFFER_BYTES {
            tracing::warn!(
                bytes = chunk.len(),
                "sse: chunk exceeds buffer cap; discarding"
            );
            return Vec::new();
        }
        // Protect against unbounded buffering: if the buffer would grow
        // past the cap without a newline terminator, discard the buffered
        // data rather than consuming unbounded memory. The event-data
        // accounting is reset with it, or a stale count would spuriously
        // discard later valid events.
        if self.buffer.len() + chunk.len() > MAX_BUFFER_BYTES {
            tracing::warn!("sse: buffer cap exceeded; discarding buffered data");
            self.buffer.clear();
            self.data_lines.clear();
            self.data_lines_bytes = 0;
        }
        self.buffer.extend_from_slice(chunk);

        let mut payloads = Vec::new();
        // Process lines with a cursor and drain the consumed prefix once,
        // after the loop: drain(..=newline) per line would shift the rest
        // of the buffer on EVERY iteration (O(n²) for many-line chunks).
        let mut consumed = 0usize;
        while let Some(rel) = self.buffer[consumed..].iter().position(|&b| b == b'\n') {
            let end = consumed + rel;
            let mut line = &self.buffer[consumed..end]; // strip '\n'
            consumed = end + 1;
            line = match line.strip_suffix(b"\r") {
                Some(stripped) => stripped,
                None => line,
            };
            if line.is_empty() {
                // Blank line terminates the event.
                if !self.data_lines.is_empty() {
                    // Drop the trailing separator added after the last
                    // data line.
                    self.data_lines.pop();
                    payloads.push(String::from_utf8_lossy(&self.data_lines).into_owned());
                    self.data_lines.clear();
                    self.data_lines_bytes = 0;
                }
                continue;
            }
            if let Some(data) = line.strip_prefix(b"data:") {
                let data = data.strip_prefix(b" ").unwrap_or(data);
                if let Ok(s) = std::str::from_utf8(data) {
                    // The cap must also bound events built from MANY
                    // newline-terminated lines without a blank line
                    // between them.
                    self.data_lines_bytes += s.len() + 1;
                    if self.data_lines_bytes > MAX_BUFFER_BYTES {
                        tracing::warn!(
                            bytes = self.data_lines_bytes,
                            "sse: event data exceeds buffer cap; discarding event"
                        );
                        self.data_lines.clear();
                        self.data_lines_bytes = 0;
                    } else {
                        self.data_lines.extend_from_slice(s.as_bytes());
                        self.data_lines.push(b'\n');
                    }
                } else {
                    // SSE mandates UTF-8; surface protocol violations
                    // instead of silently truncating the client stream.
                    self.invalid_utf8_lines += 1;
                    if self.invalid_utf8_lines <= 3 || self.invalid_utf8_lines % 1000 == 0 {
                        tracing::warn!(
                            count = self.invalid_utf8_lines,
                            "sse: dropping non-UTF-8 data line (protocol violation)"
                        );
                    }
                }
            }
            // Other SSE fields (`event:`, `id:`, `retry:`) are ignored for
            // conversion purposes.
        }
        if consumed > 0 {
            self.buffer.drain(..consumed);
        }
        payloads
    }

    /// Flush any remaining partial event data.
    pub fn finish(&mut self) -> Vec<String> {
        let payloads = if self.data_lines.is_empty() {
            Vec::new()
        } else {
            // Drop the trailing separator added after the last line.
            let mut buf = self.data_lines.clone();
            buf.pop();
            vec![String::from_utf8_lossy(&buf).into_owned()]
        };
        self.data_lines.clear();
        self.data_lines_bytes = 0;
        self.buffer.clear();
        payloads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_chunk_with_many_lines_stays_linear() {
        // One chunk holding thousands of newline-terminated lines exercises
        // the offset-based loop (the old per-line drain was O(n²)); the
        // payload must still be the joined data lines.
        let mut chunk = Vec::new();
        for i in 0..50_000 {
            chunk.extend_from_slice(format!("data: line{i}\n").as_bytes());
        }
        chunk.extend_from_slice(b"\n");
        let payloads = SseFraming::new().push(&chunk);
        assert_eq!(payloads.len(), 1);
        assert!(payloads[0].starts_with("line0\nline1\n"));
        assert!(payloads[0].ends_with("line49999"));
    }

    #[test]
    fn many_tiny_lines_stay_under_the_byte_cap() {
        // A flood of 1-byte data lines is bounded by the BYTE cap (a
        // per-line String collection would blow past it on allocator
        // overhead long before the byte count tripped).
        let mut f = SseFraming::new();
        let mut chunk = Vec::new();
        for _ in 0..70_000 {
            chunk.extend_from_slice(b"data: x\n");
        }
        chunk.extend_from_slice(b"\n");
        let payloads = f.push(&chunk);
        // 70k * ~9 bytes << cap, valid single event.
        assert_eq!(payloads.len(), 1);
        assert!(payloads[0].ends_with("x\nx"));
    }

    #[test]
    fn parses_single_event_across_chunks() {
        let mut f = SseFraming::new();
        let payloads = f.push(b"data: {\"a\":1}\n\n");
        assert_eq!(payloads, vec!["{\"a\":1}"]);
    }

    #[test]
    fn parses_event_split_across_chunk_boundaries() {
        let mut f = SseFraming::new();
        assert!(f.push(b"data: {\"a\":1").is_empty());
        assert!(f.push(b",\"b\":2}\n").is_empty());
        let payloads = f.push(b"\nnext: x\ndata: hello\n\n");
        assert_eq!(payloads, vec!["{\"a\":1,\"b\":2}", "hello"]);
    }

    #[test]
    fn joins_multiple_data_lines() {
        let mut f = SseFraming::new();
        let payloads = f.push(b"data: part1\ndata: part2\n\n");
        assert_eq!(payloads, vec!["part1\npart2"]);
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        let mut f = SseFraming::new();
        let payloads = f.push(b": keepalive\nevent: ping\ndata: {\"type\":\"ping\"}\n\n");
        assert_eq!(payloads, vec!["{\"type\":\"ping\"}"]);
    }

    #[test]
    fn handles_done_marker_and_crlf() {
        let mut f = SseFraming::new();
        let payloads = f.push(b"data: [DONE]\r\n\r\n");
        assert_eq!(payloads, vec!["[DONE]"]);
    }

    #[test]
    fn finish_flushes_partial_event() {
        let mut f = SseFraming::new();
        assert!(f.push(b"data: {\"x\":1}\n").is_empty());
        assert_eq!(f.finish(), vec!["{\"x\":1}"]);
        assert!(f.finish().is_empty());
    }

    #[test]
    fn many_data_lines_without_blank_line_stay_bounded() {
        // A malicious upstream sending many newline-terminated `data:`
        // lines with no blank line must not grow memory unbounded: the
        // event-data cap discards the accumulated lines.
        let mut f = SseFraming::new();
        let filler = vec![b'x'; 1024 * 1024];
        for _ in 0..70 {
            let mut chunk = Vec::with_capacity(filler.len() + 8);
            chunk.extend_from_slice(b"data: ");
            chunk.extend_from_slice(&filler);
            chunk.push(b'\n');
            f.push(&chunk);
        }
        assert!(
            f.data_lines_bytes <= MAX_BUFFER_BYTES,
            "accumulated data lines must stay bounded"
        );
        // The parser remains functional and the unterminated event flushes
        // within the cap (never unbounded).
        let payloads = f.finish();
        let total: usize = payloads.iter().map(|p| p.len()).sum();
        assert!(total <= MAX_BUFFER_BYTES);
    }

    #[test]
    fn overflow_discard_resets_event_data_accounting() {
        // A stale data_lines_bytes must not survive the buffer-overflow
        // discard, or later valid events would be spuriously dropped.
        let mut f = SseFraming::new();
        let filler = vec![b'x'; 1024 * 1024];
        // Accumulate 63 MiB of event data (below the cap).
        for _ in 0..63 {
            let mut chunk = Vec::with_capacity(filler.len() + 8);
            chunk.extend_from_slice(b"data: ");
            chunk.extend_from_slice(&filler);
            chunk.push(b'\n');
            f.push(&chunk);
        }
        // Fill the unterminated-line buffer to the cap, then overflow it:
        // the discard must reset the event-data accounting too.
        for _ in 0..64 {
            f.push(&filler);
        }
        f.push(&filler); // 65 MiB buffered -> overflow branch
        assert_eq!(f.data_lines_bytes, 0, "accounting must be reset");
        // A valid event afterwards still parses (a stale 63 MiB count
        // would have discarded it). The leftover triggering chunk is
        // terminated first as a non-data line.
        f.push(b"\n");
        let payloads = f.push(b"data: ok\n\n");
        assert_eq!(payloads, vec!["ok"]);
    }

    #[test]
    fn invalid_utf8_line_does_not_break_stream() {
        let mut f = SseFraming::new();
        assert!(f.push(b"data: \xff\xfe\n\n").is_empty());
        let payloads = f.push(b"data: ok\n\n");
        assert_eq!(payloads, vec!["ok"]);
    }

    #[test]
    fn discards_oversized_unterminated_buffer() {
        let mut f = SseFraming::new();
        // Push more than the cap without a newline: must be discarded, and a
        // following well-formed event must still parse.
        let big = vec![b'x'; MAX_BUFFER_BYTES + 1];
        assert!(f.push(&big).is_empty());
        let payloads = f.push(b"\ndata: {\"ok\":true}\n\n");
        assert_eq!(payloads, vec!["{\"ok\":true}"]);
        // No leftover garbage from the discarded run.
        assert!(f.finish().is_empty());
    }

    #[test]
    fn oversized_single_chunk_is_not_buffered() {
        let mut f = SseFraming::new();
        let big = vec![b'x'; MAX_BUFFER_BYTES + 1];
        assert!(f.push(&big).is_empty());
        // The chunk must not have been copied into the buffer.
        assert!(f.buffer.is_empty());
        let payloads = f.push(b"data: ok\n\n");
        assert_eq!(payloads, vec!["ok"]);
    }

    #[test]
    fn invalid_utf8_data_line_is_dropped_and_stream_continues() {
        let mut f = SseFraming::new();
        assert!(f.push(b"data: \xff\xfe\n\n").is_empty());
        let payloads = f.push(b"data: ok\n\n");
        assert_eq!(payloads, vec!["ok"]);
    }
}
