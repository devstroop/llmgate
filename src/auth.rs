//! Client request authentication.
//!
//! When `auth.api_keys` is non-empty, every request must present a matching
//! key via one of: `Authorization: Bearer <key>`, `api-key: <key>` or
//! `x-api-key: <key>`. Empty key list disables authentication.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;

use crate::core::AppState;

pub async fn require_auth(
    state: axum::extract::State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if state.config.auth.api_keys.is_empty() {
        return next.run(request).await;
    }
    let presented = extract_key(request.headers());
    // Compare constant-time so timing does not leak whether a prefix of the
    // key is valid.
    let key_ok = match &presented {
        Some(key) => state.config.auth.api_keys.iter().any(|k| ct_eq(k, key)),
        None => false,
    };
    if key_ok {
        next.run(request).await
    } else {
        unauthorized()
    }
}

/// Constant-time equality over two strings: returns as soon as a differing
/// byte is found but always reads at least the length of `b`, so runtime
/// does not disclose `a`'s length/prefix structure.
fn ct_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    let (aa, ab) = (a.as_bytes(), b.as_bytes());
    for i in 0..aa.len() {
        diff |= aa[i] ^ ab[i];
    }
    diff == 0
}

/// Extract the presented API key from common auth headers.
pub fn extract_key(headers: &axum::http::HeaderMap) -> Option<String> {
    for name in ["authorization", "api-key", "x-api-key"] {
        if let Some(value) = headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
        {
            if let Some(rest) = value.strip_prefix("Bearer") {
                let key = rest.trim_start();
                if !key.is_empty() {
                    return Some(key.to_string());
                }
            } else if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn unauthorized() -> Response {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header("content-type", "application/json")
        .body(Body::from(
            "{\"error\":{\"message\":\"invalid or missing api key\"}}",
        ))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn extracts_bearer_key() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer sk-123"));
        assert_eq!(extract_key(&h), Some("sk-123".to_string()));
    }

    #[test]
    fn extracts_plain_keys() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("sk-123"));
        assert_eq!(extract_key(&h), Some("sk-123".to_string()));

        let mut h = axum::http::HeaderMap::new();
        h.insert("api-key", HeaderValue::from_static("sk-123"));
        assert_eq!(extract_key(&h), Some("sk-123".to_string()));
    }

    #[test]
    fn no_key_when_absent() {
        assert_eq!(extract_key(&axum::http::HeaderMap::new()), None);
    }

    #[test]
    fn empty_bearer_is_none() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer "));
        assert_eq!(extract_key(&h), None);
    }
}
