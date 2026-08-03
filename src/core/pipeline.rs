use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::Response as AxumResponse;

use super::error::AdapterError;
use super::registry::ProtocolRegistry;
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

    if neutral.stream {
        let err = AdapterError::Internal("streaming not implemented yet".to_string());
        return error_json_response(client.serialize_error(&err));
    }

    neutral.model = state.resolver.resolve(&neutral.model);

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
