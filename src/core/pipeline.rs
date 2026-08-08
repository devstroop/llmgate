use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::Response as AxumResponse;
use futures_util::{Stream, StreamExt};

use super::error::AdapterError;
use super::neutral::{ContentBlock, NeutralResponse, NeutralStreamEvent};
use super::privacy::{RedactionEngine, RedactionSession, StreamRestorer};
use super::registry::{ProtocolAdapter, ProtocolRegistry};
use super::sse::SseFraming;
use crate::config::Config;
use crate::proxy;
use crate::resolver::ModelResolver;

/// Idle bound for streaming upstream reads: no total timeout (long SSE
/// streams must not be cut off), but an upstream that stops sending bytes
/// entirely must not hold the connection forever. Any upstream bytes
/// (including SSE keep-alive comments) reset the timer.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);
/// Cap on upstream response bodies (same as the inbound request cap).
const MAX_UPSTREAM_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Shared service state: config, registered adapters, model resolver, HTTP client.
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub registry: ProtocolRegistry,
    pub resolver: ModelResolver,
    pub http: reqwest::Client,
    /// Compiled privacy guard engine; `None` when the feature is disabled.
    pub privacy: Option<Arc<RedactionEngine>>,
}

impl AppState {
    /// Build service state. Fails when the privacy guard is enabled but
    /// its configuration is invalid — the feature fails closed rather
    /// than silently running unredacted.
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let http = proxy::client();
        let privacy = if config.privacy.enabled {
            Some(RedactionEngine::new(&config.privacy)?)
        } else {
            None
        };
        Ok(Self {
            resolver: ModelResolver::new(
                config.models.default.clone(),
                config.models.map.clone(),
                config.models.prefixes.clone(),
            ),
            registry: ProtocolRegistry::new(),
            http,
            config,
            privacy,
        })
    }
}

/// Generic conversation handler. Mounted for every inbound endpoint of every
/// configured client protocol. Reads the client-protocol request body,
/// converts to the neutral model, resolves the model name, serializes for the
/// upstream protocol, forwards, then converts the response back.
pub async fn handle_conversation(
    state: Arc<AppState>,
    client_name: String,
    request: Request<Body>,
) -> AxumResponse {
    let client = match state.registry.get(&client_name) {
        Some(a) => a,
        None => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("client protocol not registered: {client_name}"),
            );
        }
    };
    let upstream = match state.registry.get(&state.config.upstream.protocol) {
        Some(a) => a,
        None => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!(
                    "upstream protocol not registered: {}",
                    state.config.upstream.protocol
                ),
            );
        }
    };

    let body = match read_body(request).await {
        Ok(b) => b,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    let mut neutral = match client.parse_request(&body) {
        Ok(n) => n,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    neutral.model = state.resolver.resolve(&neutral.model);
    let stream = neutral.stream;

    // Privacy Guard: reversibly redact the request before it leaves the
    // gateway. The session vault lives exactly as long as the request:
    // it is moved into the response path and dropped once the response
    // has been fully restored. Redaction failure (e.g. the session token
    // cap) rejects the request — never forward unredacted data upstream.
    let privacy_session = match &state.privacy {
        Some(engine) => {
            let session = RedactionSession::new(engine.clone());
            if let Err(e) = session.redact_request(&mut neutral) {
                return error_json_response(client.serialize_error(&e));
            }
            let tokens = session.token_count();
            if tokens > 0 {
                tracing::info!(tokens, "privacy guard: request redacted");
            }
            Some(Arc::new(session))
        }
        None => None,
    };

    let upstream_body = match upstream.serialize_request(&neutral) {
        Ok(s) => s,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    let url = if stream {
        match upstream.stream_conversation_url(&state.config.upstream.url, &neutral.model) {
            Ok(u) => u,
            Err(e) => return error_json_response(client.serialize_error(&e)),
        }
    } else {
        match upstream.conversation_url(&state.config.upstream.url, &neutral.model) {
            Ok(u) => u,
            Err(e) => return error_json_response(client.serialize_error(&e)),
        }
    };
    let mut headers: Vec<(String, String)> = Vec::new();
    if !state.config.upstream.authorization.is_empty() {
        headers.push((
            "authorization".to_string(),
            state.config.upstream.authorization.clone(),
        ));
    }
    for h in &state.config.upstream.extra_headers {
        if !h.name.is_empty() {
            headers.push((h.name.clone(), h.value.clone()));
        }
    }
    headers.extend(upstream.request_headers());

    let resp = if stream {
        // Streaming: no total per-request timeout so long SSE streams are
        // not cut off (lifetime bounded by the SSE protocol itself).
        match proxy::forward_stream(&state.http, &url, &headers, &upstream_body).await {
            Ok(r) => r,
            Err(e) => {
                let err = AdapterError::Internal(format!("upstream request failed: {e}"));
                return error_json_response(client.serialize_error(&err));
            }
        }
    } else {
        let timeout = Duration::from_millis(state.config.upstream.timeout_ms);
        match proxy::forward(&state.http, &url, &headers, &upstream_body, timeout).await {
            Ok(r) => r,
            Err(e) => {
                let err = AdapterError::Internal(format!("upstream request failed: {e}"));
                return error_json_response(client.serialize_error(&err));
            }
        }
    };

    if stream {
        return handle_streaming_response(client, upstream, resp, privacy_session).await;
    }

    let status = resp.status();
    let resp_text = match read_upstream_text(resp).await {
        Ok(t) => t,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    if !status.is_success() {
        // The upstream error body may echo the rejected (redacted)
        // request; restore any minted tokens so the client sees its own
        // values, not `<EMAIL_1>` placeholders.
        let mut error_body = resp_text;
        if let Some(session) = &privacy_session {
            error_body = session.restore_text(&error_body);
        }
        let err = upstream.parse_upstream_error(status.as_u16(), &error_body);
        return error_json_response(client.serialize_error(&err));
    }

    let mut neutral_resp = match upstream.parse_response(&resp_text) {
        Ok(r) => r,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    if let Some(session) = &privacy_session {
        session.restore_response(&mut neutral_resp);
    }

    let client_body = match client.serialize_response(&neutral_resp) {
        Ok(s) => s,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(client_body))
        .unwrap()
}

async fn read_body(request: Request<Body>) -> Result<String, AdapterError> {
    const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
    let bytes = axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES)
        .await
        .map_err(|e| AdapterError::InvalidRequest(format!("failed to read body: {e}")))?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| AdapterError::InvalidRequest("body is not valid utf-8".to_string()))
}

/// Generic model-listing handler. Fetches the upstream model list, parses it
/// with the upstream adapter, and re-serializes it in the client protocol's
/// native shape.
pub async fn handle_models(state: Arc<AppState>, client_name: String) -> AxumResponse {
    let client = match state.registry.get(&client_name) {
        Some(a) => a,
        None => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("client protocol not registered: {client_name}"),
            );
        }
    };
    let upstream = match state.registry.get(&state.config.upstream.protocol) {
        Some(a) => a,
        None => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!(
                    "upstream protocol not registered: {}",
                    state.config.upstream.protocol
                ),
            );
        }
    };
    let Some(models_path) = upstream.models_path() else {
        let err = AdapterError::Internal(format!(
            "upstream protocol {} has no model listing",
            upstream.name()
        ));
        return error_json_response(client.serialize_error(&err));
    };

    let url = format!(
        "{}{}",
        state.config.upstream.url.trim_end_matches('/'),
        models_path
    );
    let mut headers: Vec<(String, String)> = Vec::new();
    if !state.config.upstream.authorization.is_empty() {
        headers.push((
            "authorization".to_string(),
            state.config.upstream.authorization.clone(),
        ));
    }
    for h in &state.config.upstream.extra_headers {
        if !h.name.is_empty() {
            headers.push((h.name.clone(), h.value.clone()));
        }
    }
    headers.extend(upstream.request_headers());

    let resp = match proxy::forward_get(
        &state.http,
        &url,
        &headers,
        Duration::from_millis(state.config.upstream.timeout_ms),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            let err = AdapterError::Internal(format!("upstream request failed: {e}"));
            return error_json_response(client.serialize_error(&err));
        }
    };

    let status = resp.status();
    let resp_text = match read_upstream_text(resp).await {
        Ok(t) => t,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };
    if !status.is_success() {
        let err = upstream.parse_upstream_error(status.as_u16(), &resp_text);
        return error_json_response(client.serialize_error(&err));
    }

    let models = match upstream.parse_models(&resp_text) {
        Ok(m) => m,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };
    let body = match client.serialize_models(&models) {
        Ok(b) => b,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

/// Generic token-counting handler. Estimates input tokens from the request
/// body using a heuristic and returns the estimate in the client protocol's
/// shape (Anthropic: `{"input_tokens": N}`).
///
/// NOTE: the response shape is Anthropic's by construction — only the
/// anthropic adapter registers `EndpointKind::CountTokens` today. If a
/// second protocol ever registers it, the shape needs adapter-level
/// serialization.
pub async fn handle_count_tokens(
    state: Arc<AppState>,
    client_name: String,
    request: Request<Body>,
) -> AxumResponse {
    let client = match state.registry.get(&client_name) {
        Some(a) => a,
        None => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("client protocol not registered: {client_name}"),
            );
        }
    };
    let body = match read_body(request).await {
        Ok(b) => b,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };
    let estimated = estimate_tokens(&body);
    let response_body = serde_json::json!({ "input_tokens": estimated }).to_string();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(response_body))
        .unwrap()
}

/// Heuristic token estimate: chars/4 + words/8.
pub fn estimate_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    let words = text.split_whitespace().count() as u64;
    chars / 4 + words / 8
}

fn error_response(status: StatusCode, message: &str) -> AxumResponse {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(format!(
            "{{\"error\": {{\"message\": {message:?}}}}}"
        )))
        .unwrap()
}

/// Render an adapter error as an HTTP response using the client protocol's
/// native error shape.
fn error_json_response((status, body): (u16, String)) -> AxumResponse {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

/// Relay a streaming upstream response, converting SSE chunks on the fly:
/// upstream SSE → `StreamDecoder` → neutral events → `StreamEncoder` →
/// client SSE lines. When a privacy session is present, token deltas are
/// restored before encoding.
async fn handle_streaming_response(
    client: Arc<dyn ProtocolAdapter>,
    upstream: Arc<dyn ProtocolAdapter>,
    resp: reqwest::Response,
    privacy: Option<Arc<RedactionSession>>,
) -> AxumResponse {
    let status = resp.status();
    if !status.is_success() {
        let resp_text = match read_upstream_text(resp).await {
            Ok(t) => t,
            Err(e) => return error_json_response(client.serialize_error(&e)),
        };
        // Same token restore as the non-streaming error path: an upstream
        // error body may echo the redacted request.
        let mut error_body = resp_text;
        if let Some(session) = &privacy {
            error_body = session.restore_text(&error_body);
        }
        let err = upstream.parse_upstream_error(status.as_u16(), &error_body);
        return error_json_response(client.serialize_error(&err));
    }

    // If the upstream responded with plain JSON despite a streaming request
    // (some OpenAI-compatible servers ignore `stream: true`), convert the
    // body into a single-event stream rather than handing the client an empty
    // stream.
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let looks_like_json = content_type
        .as_deref()
        .is_some_and(|ct| ct.starts_with("application/json"));

    if looks_like_json {
        let resp_text = match read_upstream_text(resp).await {
            Ok(t) => t,
            Err(e) => return error_json_response(client.serialize_error(&e)),
        };
        return handle_json_as_stream(client, upstream, &resp_text, privacy).await;
    }

    let upstream_stream = resp.bytes_stream();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut framing = SseFraming::new();
        let mut decoder = upstream.stream_decoder();
        let mut encoder = client.stream_encoder();
        let mut restorer = privacy.map(StreamRestorer::new);

        let mut running = true;
        // MessageStop events (from normal payloads and from decoder.finish())
        // are buffered and encoded only AFTER the privacy restorer's held
        // tail is flushed — never before it, or the client receives content
        // after the terminal event.
        let mut terminal_events: Vec<NeutralStreamEvent> = Vec::new();
        // Set when the upstream connection failed or stalled: the stream
        // must end WITHOUT a terminal marker so clients treat it as
        // truncated, not as a successful completion.
        let mut upstream_failed = false;
        // Set when an Error event was encoded and delivered: also suppress
        // the terminal marker after an explicit upstream error.
        let mut saw_error_event = false;
        let mut stream = Box::pin(upstream_stream);
        while running {
            // No total timeout (long SSE streams must not be cut off), but
            // an idle upstream that stops sending bytes must not hold the
            // connection forever: bound each chunk read. Also abort as soon
            // as the client disconnects (receiver dropped) instead of
            // lingering on the upstream connection until the idle timeout.
            tokio::select! {
                _ = tx.closed() => {
                    running = false;
                }
                chunk = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next()) => {
                    match chunk {
                        Ok(Some(Ok(bytes))) => {
                            for payload in framing.push(&bytes) {
                                if payload == "[DONE]" {
                                    running = false;
                                    break;
                                }
                                if !process_payload(
                                    &payload,
                                    &mut *decoder,
                                    &mut *encoder,
                                    restorer.as_mut(),
                                    &tx,
                                    &mut saw_error_event,
                                    &mut terminal_events,
                                )
                                .await
                                {
                                    running = false;
                                    break;
                                }
                            }
                        }
                        Ok(Some(Err(e))) => {
                            tracing::warn!("upstream stream error: {e}");
                            upstream_failed = true;
                            running = false;
                        }
                        Ok(None) => {
                            running = false;
                        }
                        Err(_) => {
                            tracing::warn!(
                                "upstream stream idle timeout ({STREAM_IDLE_TIMEOUT:?}); closing stream"
                            );
                            upstream_failed = true;
                            running = false;
                        }
                    }
                }
            }
        }

        if !tx.is_closed() {
            for payload in framing.finish() {
                // The stream-end marker is not a data payload; it must not
                // be fed to the decoder (which would try to parse it as
                // JSON).
                if payload == "[DONE]" {
                    break;
                }
                if !process_payload(
                    &payload,
                    &mut *decoder,
                    &mut *encoder,
                    restorer.as_mut(),
                    &tx,
                    &mut saw_error_event,
                    &mut terminal_events,
                )
                .await
                {
                    break;
                }
            }
            // Terminal flush ordering: decoder.finish() events first (they
            // may carry held-back content that needs restoring), through
            // the restorer; then the restorer's own held plain text; then
            // the terminal event. MessageStop events are held until the
            // content flushes have been emitted.
            if !upstream_failed {
                for event in decoder.finish() {
                    match restorer.as_mut() {
                        Some(r) => match r.restore_event(event) {
                            Some(e) => {
                                if matches!(e, NeutralStreamEvent::MessageStop { .. }) {
                                    terminal_events.push(e);
                                } else if !send_lines(encoder.encode(e), &tx).await {
                                    break;
                                }
                            }
                            None => continue,
                        },
                        None => {
                            if matches!(event, NeutralStreamEvent::MessageStop { .. }) {
                                terminal_events.push(event);
                            } else if !send_lines(encoder.encode(event), &tx).await {
                                break;
                            }
                        }
                    }
                }
            }
            // Flush held partial tokens before the terminal event so a
            // stream that ended mid-token still delivers the plain text.
            if let Some(restorer) = restorer.as_mut() {
                for event in restorer.finish() {
                    if !send_lines(encoder.encode(event), &tx).await {
                        break;
                    }
                }
            }
            // On a transport failure, do not synthesize a normal terminal
            // event: the client must see an incomplete stream. Each
            // decoder's finish() is called exactly once.
            if !upstream_failed {
                for event in terminal_events {
                    if !send_lines(encoder.encode(event), &tx).await {
                        break;
                    }
                }
                if !saw_error_event {
                    let _ = tx.send(Ok(axum::body::Bytes::from(encoder.done()))).await;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        // Prevent intermediate proxies (e.g. nginx) from buffering the SSE
        // stream, which would destroy event liveness.
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(keepalive_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
            Duration::from_secs(15),
        )))
        .unwrap()
}

/// Wrap a stream of SSE bytes with periodic keep-alive comment lines sent
/// during idle gaps, so long-lived streams (thinking, tool calls) survive
/// proxies and clients that drop idle connections.
fn keepalive_stream<S>(
    inner: S,
    interval: Duration,
) -> impl futures_util::Stream<Item = Result<axum::body::Bytes, std::io::Error>>
where
    S: Stream<Item = Result<axum::body::Bytes, std::io::Error>> + 'static,
{
    let inner = Box::pin(inner);
    // First keep-alive only after the idle period: `interval()` completes
    // its first tick immediately, which would inject a spurious leading
    // comment before any real event.
    let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let keepalive = axum::body::Bytes::from(": keepalive\n\n");

    // The pinned upstream, the interval, and the keep-alive byte string live
    // in the *state* carried between unfold iterations (not captured by the
    // FnMut), so the closure can be called many times.
    futures_util::stream::unfold(
        (inner, tick, keepalive),
        move |(mut inner, mut tick, keepalive)| async move {
            // Each iteration yields exactly one item (a real event or a
            // keep-alive), never looping within a single poll.
            tokio::select! {
                next = inner.next() => {
                                next.map(|item| (item, (inner, tick, keepalive)))
                            },
                _ = tick.tick() => Some((Ok(keepalive.clone()), (inner, tick, keepalive))),
            }
        },
    )
}

/// The upstream replied with a plain JSON body despite a streaming request.
/// Convert it into a single, already-complete stream for the client. Two
/// shapes are handled: chunk-shaped JSON (a stream stripped of its SSE
/// framing — parsed by the stream decoder) and a standard non-stream body
/// (`message`-shaped — parsed as a response and synthesized into events).
/// Either way the privacy restorer runs over the events, and the client
/// receives the terminal line. This avoids handing a client that requested
/// streaming an empty stream.
async fn handle_json_as_stream(
    client: Arc<dyn ProtocolAdapter>,
    upstream: Arc<dyn ProtocolAdapter>,
    body: &str,
    privacy: Option<Arc<RedactionSession>>,
) -> AxumResponse {
    let mut encoder = client.stream_encoder();
    let mut restorer = privacy.map(StreamRestorer::new);
    let mut lines: Vec<String> = Vec::new();

    // First try the stream decoder: some servers answer `stream: true` with
    // chunk-shaped JSON, which the decoder parses natively.
    let mut events: Vec<NeutralStreamEvent> = Vec::new();
    {
        let mut decoder = upstream.stream_decoder();
        events.extend(decoder.feed(body));
        events.extend(decoder.finish());
    }
    let has_content = events.iter().any(|e| {
        matches!(
            e,
            NeutralStreamEvent::TextDelta(_)
                | NeutralStreamEvent::ReasoningDelta(_)
                | NeutralStreamEvent::ToolCallDelta { .. }
        )
    });
    if !has_content {
        // Standard non-stream body: stream decoders only read delta/SSE
        // shapes, so parse the response and synthesize a single-event
        // stream instead of handing the client an empty one.
        let resp = match upstream.parse_response(body) {
            Ok(r) => r,
            Err(e) => return error_json_response(client.serialize_error(&e)),
        };
        events = response_to_events(resp);
    }

    // Mirror the streaming path's terminal semantics: after an error
    // event that was actually delivered, do not append the done marker —
    // the client must see the stream as failed, not completed. MessageStop
    // is buffered and emitted only after the restorer's held tail is
    // flushed (never before it, or content would follow the terminal).
    let mut saw_error = false;
    let mut terminal_events: Vec<NeutralStreamEvent> = Vec::new();
    for event in events {
        let event = match restorer.as_mut() {
            Some(r) => match r.restore_event(event) {
                Some(e) => e,
                None => continue,
            },
            None => event,
        };
        if matches!(event, NeutralStreamEvent::MessageStop { .. }) {
            terminal_events.push(event);
            continue;
        }
        let is_error = matches!(&event, NeutralStreamEvent::Error(_));
        let encoded = encoder.encode(event);
        if is_error && !encoded.is_empty() {
            saw_error = true;
        }
        lines.extend(encoded);
    }
    if let Some(restorer) = restorer.as_mut() {
        for event in restorer.finish() {
            lines.extend(encoder.encode(event));
        }
    }
    for event in terminal_events {
        lines.extend(encoder.encode(event));
    }
    if !saw_error {
        let done = encoder.done();
        if !done.is_empty() {
            lines.push(done);
        }
    }

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from(lines.join("")))
        .unwrap()
}

/// Read an upstream response body with a hard size cap, so a broken or
/// compromised upstream cannot exhaust gateway memory (the client request
/// path has the same 16 MiB cap).
async fn read_upstream_text(resp: reqwest::Response) -> Result<String, AdapterError> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|e| AdapterError::Internal(format!("upstream body read failed: {e}")))?;
        if bytes.len().saturating_add(chunk.len()) > MAX_UPSTREAM_BODY_BYTES {
            return Err(AdapterError::Internal(format!(
                "upstream response body exceeds {MAX_UPSTREAM_BODY_BYTES} bytes"
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes)
        .map_err(|_| AdapterError::Internal("upstream response is not valid utf-8".into()))
}

/// Feed one SSE payload through the decoder, restoring any privacy tokens
/// and encoding resulting events for the client. Returns false if the
/// client went away. Sets `saw_error` when an Error event was encoded.
async fn process_payload(
    payload: &str,
    decoder: &mut dyn crate::core::registry::StreamDecoder,
    encoder: &mut dyn crate::core::registry::StreamEncoder,
    mut restorer: Option<&mut StreamRestorer>,
    tx: &tokio::sync::mpsc::Sender<Result<axum::body::Bytes, std::io::Error>>,
    saw_error: &mut bool,
    terminal: &mut Vec<NeutralStreamEvent>,
) -> bool {
    for event in decoder.feed(payload) {
        let event = match restorer.as_deref_mut() {
            Some(r) => match r.restore_event(event) {
                Some(e) => e,
                None => continue,
            },
            None => event,
        };
        // A MessageStop inside a normal payload must NOT be encoded yet:
        // the privacy restorer may hold a text tail (text following the
        // last '<') that is only flushed by restorer.finish() — encoding
        // the terminal first would deliver content after protocol end.
        if matches!(event, NeutralStreamEvent::MessageStop { .. }) {
            terminal.push(event);
            continue;
        }
        // Only flag the error once the event has actually been restored
        // AND produced output: if the restorer filtered it or the encoder
        // emitted nothing, the terminal marker must not be suppressed for
        // an error the client never saw.
        let is_error = matches!(&event, NeutralStreamEvent::Error(_));
        let lines = encoder.encode(event);
        if is_error && !lines.is_empty() {
            *saw_error = true;
        }
        if !send_lines(lines, tx).await {
            return false;
        }
    }
    true
}

async fn send_lines(
    lines: Vec<String>,
    tx: &tokio::sync::mpsc::Sender<Result<axum::body::Bytes, std::io::Error>>,
) -> bool {
    if lines.is_empty() {
        return true;
    }
    let joined = lines.join("");
    tx.send(Ok(axum::body::Bytes::from(joined))).await.is_ok()
}

/// Convert a parsed non-stream response into the event sequence of a
/// single-event stream (the JSON-as-stream fallback): MessageStart, one
/// delta per content block, MessageStop.
fn response_to_events(resp: NeutralResponse) -> Vec<NeutralStreamEvent> {
    let mut events = vec![NeutralStreamEvent::MessageStart {
        id: resp.id,
        model: resp.model,
        usage: None,
    }];
    // Tool-call indices must be contiguous per protocol: a separate
    // counter increments only for ToolUse blocks (the content-block
    // position would leave holes when text/thinking blocks precede).
    let mut tool_index = 0u32;
    for block in resp.content.iter() {
        match block {
            ContentBlock::Text(t) => events.push(NeutralStreamEvent::TextDelta(t.clone())),
            ContentBlock::Thinking { thinking, .. } => {
                events.push(NeutralStreamEvent::ReasoningDelta(thinking.clone()))
            }
            ContentBlock::ToolUse { id, name, input } => {
                events.push(NeutralStreamEvent::ToolCallDelta {
                    index: tool_index,
                    id: id.clone(),
                    name: name.clone(),
                    arguments: input.to_string(),
                });
                tool_index += 1;
            }
            // Image blocks and tool results do not map to stream deltas;
            // dropping them silently would hand the client an incomplete
            // stream, so make it observable.
            ContentBlock::Image { .. } => {
                tracing::warn!(
                    "json-as-stream fallback: dropping image block (not representable in a stream)"
                );
            }
            ContentBlock::RedactedThinking { .. } => {
                tracing::warn!(
                    "json-as-stream fallback: dropping redacted-thinking block (not representable in a stream)"
                );
            }
            ContentBlock::ToolResult { .. } => {
                tracing::warn!(
                    "json-as-stream fallback: dropping tool-result block (not representable in a stream)"
                );
            }
        }
    }
    events.push(NeutralStreamEvent::MessageStop {
        finish_reason: resp.finish_reason,
        usage: resp.usage,
    });
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::neutral::{FinishReason, NeutralUsage};

    #[test]
    fn response_to_events_maps_blocks_to_deltas() {
        let resp = NeutralResponse {
            id: "r1".into(),
            model: "m".into(),
            content: vec![
                ContentBlock::Text("hello".into()),
                ContentBlock::Thinking {
                    thinking: "hmm".into(),
                    signature: None,
                },
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "f".into(),
                    input: serde_json::json!({"a": 1}),
                },
            ],
            finish_reason: FinishReason::Stop,
            usage: Some(NeutralUsage {
                input_tokens: 3,
                output_tokens: 7,
            }),
        };
        let events = response_to_events(resp);
        assert_eq!(
            events[0],
            NeutralStreamEvent::MessageStart {
                id: "r1".into(),
                model: "m".into(),
                usage: None
            }
        );
        assert_eq!(events[1], NeutralStreamEvent::TextDelta("hello".into()));
        assert_eq!(events[2], NeutralStreamEvent::ReasoningDelta("hmm".into()));
        // The tool-call index is contiguous (0), NOT the content-block
        // position (2): text/thinking blocks before the call must not
        // leave holes in the client's tool_calls array.
        assert!(matches!(
            &events[3],
            NeutralStreamEvent::ToolCallDelta { index, id, name, arguments }
                if *index == 0 && id == "call_1" && name == "f" && arguments.contains("a")
        ));
        assert!(matches!(
            &events[4],
            NeutralStreamEvent::MessageStop { finish_reason: FinishReason::Stop, usage: Some(u) }
                if u.input_tokens == 3 && u.output_tokens == 7
        ));
    }

    #[test]
    fn response_to_events_empty_content_still_terminates() {
        let resp = NeutralResponse {
            id: String::new(),
            model: String::new(),
            content: vec![],
            finish_reason: FinishReason::Length,
            usage: None,
        };
        let events = response_to_events(resp);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[1],
            NeutralStreamEvent::MessageStop {
                finish_reason: FinishReason::Length,
                usage: None
            }
        ));
    }
}
