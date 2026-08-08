//! Gemini (Google AI Studio / Vertex `generateContent`) protocol adapter.
//!
//! Converts between the Gemini `generateContent` wire format and the neutral
//! model. Primarily an upstream adapter: OpenAI- or Anthropic-format clients
//! can be routed to Gemini through this gateway. Client-side Gemini inbound
//! (`/v1beta/models/{model}:generateContent`) needs path-parameterized
//! routing, which is a follow-up milestone.

mod convert;
mod stream;

use std::sync::Arc;

use serde_json::{Value, json};

use crate::core::error::AdapterError;
use crate::core::neutral::{ModelInfo, NeutralRequest, NeutralResponse};
use crate::core::registry::{EndpointKind, ProtocolAdapter, StreamDecoder, StreamEncoder};

pub use convert::{
    parse_request, parse_response, serialize_error, serialize_request, serialize_response,
};
pub use stream::{GeminiStreamDecoder, GeminiStreamEncoder};

/// Gemini `generateContent` protocol adapter.
pub struct GeminiAdapter;

impl Default for GeminiAdapter {
    fn default() -> Self {
        Self
    }
}

impl ProtocolAdapter for GeminiAdapter {
    fn name(&self) -> &'static str {
        "gemini"
    }

    fn endpoints(&self) -> Vec<(&'static str, EndpointKind)> {
        // Gemini client inbound paths are parameterized (`/v1beta/models/
        // {model}:generateContent`); not yet supported by the static router.
        // This adapter is upstream-side only for now.
        Vec::new()
    }

    fn conversation_url(&self, base: &str, model: &str) -> Result<String, AdapterError> {
        validate_model_segment(model)?;
        Ok(format!(
            "{}/v1beta/{}:generateContent",
            base.trim_end_matches('/'),
            resource_path(model)
        ))
    }

    fn stream_conversation_url(&self, base: &str, model: &str) -> Result<String, AdapterError> {
        validate_model_segment(model)?;
        Ok(format!(
            "{}/v1beta/{}:streamGenerateContent?alt=sse",
            base.trim_end_matches('/'),
            resource_path(model)
        ))
    }

    fn models_path(&self) -> Option<&'static str> {
        Some("/v1beta/models")
    }

    /// Gemini's model list is `{"models": [{"name": "models/<id>", ...}]}` —
    /// not the `{"data": [...]}` default. Strip the `models/` prefix from
    /// each id and use `displayName` as `owned_by`.
    fn parse_models(&self, body: &str) -> Result<Vec<ModelInfo>, AdapterError> {
        let root: Value = serde_json::from_str(body)
            .map_err(|e| AdapterError::InvalidRequest(format!("invalid JSON: {e}")))?;
        let models = root
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| AdapterError::InvalidRequest("expected a models array".to_string()))?;
        Ok(models
            .iter()
            .filter_map(|m| {
                let name = m.get("name")?.as_str()?;
                let id = name
                    .strip_prefix("models/")
                    .map(str::to_string)
                    .unwrap_or_else(|| name.to_string());
                Some(ModelInfo {
                    id,
                    owned_by: m
                        .get("displayName")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                })
            })
            .collect())
    }

    fn serialize_models(&self, models: &[ModelInfo]) -> Result<String, AdapterError> {
        let data: Vec<Value> = models
            .iter()
            .map(|m| {
                json!({
                    "name": format!("models/{}", m.id),
                    "displayName": m.owned_by,
                })
            })
            .collect();
        serde_json::to_string(&json!({ "models": data }))
            .map_err(|e| AdapterError::Internal(e.to_string()))
    }

    fn parse_request(&self, body: &str) -> Result<NeutralRequest, AdapterError> {
        parse_request(body)
    }

    fn serialize_request(&self, req: &NeutralRequest) -> Result<String, AdapterError> {
        serialize_request(req)
    }

    fn parse_response(&self, body: &str) -> Result<NeutralResponse, AdapterError> {
        parse_response(body)
    }

    fn serialize_response(&self, resp: &NeutralResponse) -> Result<String, AdapterError> {
        serialize_response(resp)
    }

    fn parse_upstream_error(&self, status: u16, body: &str) -> AdapterError {
        // Gemini error bodies: {"error": {"code": 400, "message": "...",
        // "status": "INVALID_ARGUMENT"}}. Map HTTP status onto the taxonomy.
        match status {
            400 => AdapterError::InvalidRequest(body.to_string()),
            401 => AdapterError::Authentication,
            403 => AdapterError::PermissionDenied,
            429 => AdapterError::RateLimit {
                retry_after_secs: None,
            },
            503 => AdapterError::Overloaded,
            _ => AdapterError::Upstream {
                status,
                body: body.to_string(),
            },
        }
    }

    fn serialize_error(&self, err: &AdapterError) -> (u16, String) {
        serialize_error(err)
    }

    fn stream_decoder(&self) -> Box<dyn StreamDecoder> {
        Box::new(GeminiStreamDecoder::new())
    }

    fn stream_encoder(&self) -> Box<dyn StreamEncoder> {
        Box::new(GeminiStreamEncoder::new())
    }
}

/// Map a model name to the Gemini resource path. Bare ids (e.g.
/// `gemini-2.5-flash`) live under `models/`; already-prefixed names
/// (`models/x`, `tunedModels/y`) are resource paths in their own right and
/// must not be double-prefixed.
fn resource_path(model: &str) -> String {
    if model.starts_with("models/") || model.starts_with("tunedModels/") {
        model.to_string()
    } else {
        format!("models/{model}")
    }
}

/// Validate that a client-supplied model name is safe to embed in the
/// upstream URL path. The model segment is client-controlled, so anything
/// beyond unreserved path characters (plus `/` for the `models/` and
/// `tunedModels/` prefixes) is rejected, as are `.`/`..` segments — a
/// crafted name must not be able to inject query params, fragments, or
/// path traversal into the upstream request.
fn validate_model_segment(model: &str) -> Result<(), AdapterError> {
    if model.is_empty() {
        return Err(AdapterError::InvalidRequest(
            "model name must not be empty".to_string(),
        ));
    }
    if !model
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'/'))
    {
        return Err(AdapterError::InvalidRequest(format!(
            "model name contains characters not allowed in a URL path segment: {model:?}"
        )));
    }
    for segment in model.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(AdapterError::InvalidRequest(format!(
                "model name must not contain empty or '.'/'..' path segments: {model:?}"
            )));
        }
    }
    Ok(())
}

/// Convenience for registering with the registry.
pub fn adapter() -> Arc<GeminiAdapter> {
    Arc::new(GeminiAdapter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> GeminiAdapter {
        GeminiAdapter
    }

    #[test]
    fn rejects_query_injection_in_model() {
        // A crafted model name must never inject query params into the
        // upstream URL.
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "gpt-4?x=1")
                .is_err()
        );
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "gpt-4#frag")
                .is_err()
        );
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "a&b=1")
                .is_err()
        );
    }

    #[test]
    fn rejects_path_traversal_in_model() {
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "../v1beta")
                .is_err()
        );
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "a/../b")
                .is_err()
        );
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "..")
                .is_err()
        );
        assert!(
            adapter()
                .conversation_url("https://api.example.com", ".")
                .is_err()
        );
    }

    #[test]
    fn rejects_empty_and_whitespace_models() {
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "")
                .is_err()
        );
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "gemini 2.5")
                .is_err()
        );
        assert!(
            adapter()
                .conversation_url("https://api.example.com", "gemini%2Dflash")
                .is_err()
        );
    }

    #[test]
    fn accepts_normal_model_ids() {
        let url = adapter()
            .conversation_url(
                "https://generativelanguage.googleapis.com",
                "gemini-2.5-flash",
            )
            .unwrap();
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
    }

    #[test]
    fn accepts_prefixed_model_ids() {
        // Prefixed ids are full resource paths: they must not be
        // double-prefixed with `models/`.
        let url = adapter()
            .conversation_url("https://api.example.com", "models/gemini-2.5-flash")
            .unwrap();
        assert_eq!(
            url,
            "https://api.example.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
        let url = adapter()
            .conversation_url("https://api.example.com", "tunedModels/my-tuned")
            .unwrap();
        assert_eq!(
            url,
            "https://api.example.com/v1beta/tunedModels/my-tuned:generateContent"
        );
    }

    #[test]
    fn stream_url_switches_endpoint_and_keeps_validation() {
        let url = adapter()
            .stream_conversation_url("https://api.example.com/", "gemini-2.5-pro")
            .unwrap();
        assert_eq!(
            url,
            "https://api.example.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
        assert!(
            adapter()
                .stream_conversation_url("https://api.example.com", "gemini?alt=x")
                .is_err()
        );
    }
}
