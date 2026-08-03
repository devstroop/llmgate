use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::Response as AxumResponse;
use futures_util::{Stream, StreamExt};

use super::error::AdapterError;
use super::registry::{ProtocolAdapter, ProtocolRegistry};
use super::sse::SseFraming;
use crate::config::Config;
use crate::proxy;
use crate::resolver::ModelResolver;

/// Shared service state: config, registered adapters, model resolver, HTTP client.
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub registry: ProtocolRegistry,
    pub resolver: ModelResolver,
    pub http: reqwest::Client,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let http = proxy::client();
        Self {
            resolver: ModelResolver::new(
                config.models.default.clone(),
                config.models.map.clone(),
                config.models.prefixes.clone(),
            ),
            registry: ProtocolRegistry::new(),
            http,
            config,
        }
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

    let upstream_body = match upstream.serialize_request(&neutral) {
        Ok(s) => s,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

    let url = if stream {
        upstream.stream_conversation_url(&state.config.upstream.url, &neutral.model)
    } else {
        upstream.conversation_url(&state.config.upstream.url, &neutral.model)
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
        return handle_streaming_response(client, upstream, resp).await;
    }

    let status = resp.status();
    let resp_text = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            let err = AdapterError::Internal(format!("failed to read upstream body: {e}"));
            return error_json_response(client.serialize_error(&err));
        }
    };

    if !status.is_success() {
        let err = upstream.parse_upstream_error(status.as_u16(), &resp_text);
        return error_json_response(client.serialize_error(&err));
    }

    let neutral_resp = match upstream.parse_response(&resp_text) {
        Ok(r) => r,
        Err(e) => return error_json_response(client.serialize_error(&e)),
    };

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
    let resp_text = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            let err = AdapterError::Internal(format!("failed to read upstream body: {e}"));
            return error_json_response(client.serialize_error(&err));
        }
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
/// client SSE lines.
async fn handle_streaming_response(
    client: Arc<dyn ProtocolAdapter>,
    upstream: Arc<dyn ProtocolAdapter>,
    resp: reqwest::Response,
) -> AxumResponse {
    let status = resp.status();
    if !status.is_success() {
        let resp_text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                let err = AdapterError::Internal(format!("failed to read upstream body: {e}"));
                return error_json_response(client.serialize_error(&err));
            }
        };
        let err = upstream.parse_upstream_error(status.as_u16(), &resp_text);
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
        let resp_text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                let err = AdapterError::Internal(format!("failed to read upstream body: {e}"));
                return error_json_response(client.serialize_error(&err));
            }
        };
        return handle_json_as_stream(client, upstream, &resp_text).await;
    }

    let upstream_stream = resp.bytes_stream();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut framing = SseFraming::new();
        let mut decoder = upstream.stream_decoder();
        let mut encoder = client.stream_encoder();

        let mut running = true;
        let mut stream = Box::pin(upstream_stream);
        while running {
            let chunk = stream.next().await;
            match chunk {
                Some(Ok(bytes)) => {
                    for payload in framing.push(&bytes) {
                        if payload == "[DONE]" {
                            running = false;
                            break;
                        }
                        if !process_payload(&payload, &mut *decoder, &mut *encoder, &tx).await {
                            running = false;
                            break;
                        }
                    }
                }
                Some(Err(e)) => {
                    tracing::warn!("upstream stream error: {e}");
                    let events = decoder.finish();
                    for event in events {
                        if !send_lines(encoder.encode(event), &tx).await {
                            break;
                        }
                    }
                    running = false;
                }
                None => {
                    running = false;
                }
            }
        }

        if !tx.is_closed() {
            for payload in framing.finish() {
                if !process_payload(&payload, &mut *decoder, &mut *encoder, &tx).await {
                    break;
                }
            }
            for event in decoder.finish() {
                if !send_lines(encoder.encode(event), &tx).await {
                    break;
                }
            }
            let _ = tx.send(Ok(axum::body::Bytes::from(encoder.done()))).await;
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
    let mut tick = tokio::time::interval(interval);
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
/// Convert it into a single, already-complete stream for the client: decode
/// the body through the upstream decoder, encode the neutral events with the
/// client encoder, then send the terminal line. This avoids handing a client
/// that requested streaming an empty stream.
async fn handle_json_as_stream(
    client: Arc<dyn ProtocolAdapter>,
    upstream: Arc<dyn ProtocolAdapter>,
    body: &str,
) -> AxumResponse {
    let mut decoder = upstream.stream_decoder();
    let mut encoder = client.stream_encoder();
    let mut lines: Vec<String> = Vec::new();

    for event in decoder.feed(body) {
        lines.extend(encoder.encode(event));
    }
    for event in decoder.finish() {
        lines.extend(encoder.encode(event));
    }
    let done = encoder.done();
    if !done.is_empty() {
        lines.push(done);
    }

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from(lines.join("")))
        .unwrap()
}

/// Feed one SSE payload through the decoder, encoding resulting events for
/// the client. Returns false if the client went away.
async fn process_payload(
    payload: &str,
    decoder: &mut dyn crate::core::registry::StreamDecoder,
    encoder: &mut dyn crate::core::registry::StreamEncoder,
    tx: &tokio::sync::mpsc::Sender<Result<axum::body::Bytes, std::io::Error>>,
) -> bool {
    for event in decoder.feed(payload) {
        if !send_lines(encoder.encode(event), tx).await {
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
