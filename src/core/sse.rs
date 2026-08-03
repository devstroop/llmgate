//! Minimal SSE framing parser (protocol-agnostic).
//!
//! Consumes raw byte chunks of an SSE stream and yields complete event
//! payloads: the concatenation of all `data:` lines of an event, terminated
//! by a blank line. Comment lines (`:`), `event:`, `id:`, `retry:` fields are
//! ignored; `[DONE]` and other payloads pass through verbatim.

#[derive(Default)]
pub struct SseFraming {
    buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseFraming {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns the complete event payloads found (in order).
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut payloads = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|&b| b == b'\n') {
            let line = self.buffer.drain(..=newline).collect::<Vec<u8>>();
            let line = &line[..line.len() - 1]; // strip '\n'
            let line = match line.strip_suffix(b"\r") {
                Some(stripped) => stripped,
                None => line,
            };
            if line.is_empty() {
                // Blank line terminates the event.
                if !self.data_lines.is_empty() {
                    payloads.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                }
                continue;
            }
            if let Some(data) = line.strip_prefix(b"data:") {
                let data = data.strip_prefix(b" ").unwrap_or(data);
                if let Ok(s) = String::from_utf8(data.to_vec()) {
                    self.data_lines.push(s);
                }
            }
            // Other SSE fields (`event:`, `id:`, `retry:`) and comments are
            // ignored for conversion purposes.
        }
        payloads
    }

    /// Flush any remaining partial event data.
    pub fn finish(&mut self) -> Vec<String> {
        let payloads = if self.data_lines.is_empty() {
            Vec::new()
        } else {
            vec![self.data_lines.join("\n")]
        };
        self.data_lines.clear();
        self.buffer.clear();
        payloads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
