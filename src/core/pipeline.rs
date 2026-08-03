use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::Response as AxumResponse;
use futures_util::StreamExt;

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
        let http = proxy::client(Duration::from_millis(config.upstream.timeout_ms));
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

    let url = upstream.conversation_url(&state.config.upstream.url, &neutral.model);
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

    let resp = match proxy::forward(&state.http, &url, &headers, &upstream_body).await {
        Ok(r) => r,
        Err(e) => {
            let err = AdapterError::Internal(format!("upstream request failed: {e}"));
            return error_json_response(client.serialize_error(&err));
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

    let resp = match proxy::forward_get(&state.http, &url, &headers).await {
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
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
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
