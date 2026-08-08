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
    mut request: Request<Body>,
    next: Next,
) -> Response {
    if state.config.auth.api_keys.is_empty() {
        return next.run(request).await;
    }
    let presented = match extract_key(request.headers()) {
        Some(k) => k,
        // No credential at all: never accepted while auth is enabled —
        // otherwise a configured empty-string key (e.g. `api_keys = [""]`,
        // which config validation also rejects) would accept keyless
        // requests.
        None => return unauthorized(),
    };
    // Bound the comparison by the LONGEST configured key: a credential
    // longer than every key cannot match, and must not be allowed to drive
    // the constant-time loop over attacker-chosen length.
    let max_key_len = state
        .config
        .auth
        .api_keys
        .iter()
        .map(|k| k.len())
        .max()
        .unwrap_or(0);
    if presented.len() > max_key_len {
        return unauthorized();
    }
    // Compare constant-time so timing does not leak whether a prefix of the
    // key is valid. All configured keys are compared (no early exit) so the
    // runtime does not reveal which list position matched.
    let mut key_ok = false;
    for k in &state.config.auth.api_keys {
        key_ok |= ct_eq(k, presented.as_str());
    }
    if key_ok {
        // Defense in depth: the client's gateway credential must never
        // travel upstream. Adapters build upstream requests from
        // `upstream.authorization`/`extra_headers` (server-side config),
        // not from the inbound request — but strip the credential
        // headers anyway so no downstream code path can accidentally
        // copy them into an upstream call (a redirect, a logged request,
        // or a future header-forwarding adapter would otherwise disclose
        // the gateway key to the provider or a redirect target).
        strip_credential_headers(request.headers_mut());
        next.run(request).await
    } else {
        unauthorized()
    }
}

/// Remove the credential headers the gateway accepts (`Authorization`,
/// `api-key`, `x-api-key`) so they cannot leak into the upstream request
/// or any downstream hop.
fn strip_credential_headers(headers: &mut axum::http::HeaderMap) {
    headers.remove(axum::http::header::AUTHORIZATION);
    for name in ["api-key", "x-api-key"] {
        headers.remove(name);
    }
}

/// Constant-time equality over two strings.
///
/// Both inputs are read up to their maximum length, with the length
/// difference folded into the accumulated diff, so the runtime does not
/// early-exit on a length mismatch or on the first differing byte. Timing
/// still scales with the longer input's length (inherent to comparing
/// variable-length secrets without hashing), but it reveals nothing about
/// whether the lengths match or where bytes differ.
fn ct_eq(a: &str, b: &str) -> bool {
    let (aa, ab) = (a.as_bytes(), b.as_bytes());
    let n = aa.len().max(ab.len());
    let mut diff = (aa.len() as u64) ^ (ab.len() as u64);
    for i in 0..n {
        let x = aa.get(i).copied().unwrap_or(0);
        let y = ab.get(i).copied().unwrap_or(0);
        diff |= (x ^ y) as u64;
    }
    diff == 0
}

/// Extract the presented API key from common auth headers.
///
/// `Authorization` is parsed as a scheme plus credentials (RFC 6750): only
/// a case-insensitive `Bearer` scheme is accepted, any other scheme is
/// ignored so it cannot shadow a valid `api-key`/`x-api-key` header, and
/// credentials are returned without the scheme prefix.
pub fn extract_key(headers: &axum::http::HeaderMap) -> Option<String> {
    for name in ["authorization", "api-key", "x-api-key"] {
        let Some(value) = headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
        else {
            continue;
        };
        if name == "authorization" {
            let Some((scheme, creds)) = value.split_once(char::is_whitespace) else {
                continue;
            };
            if !scheme.eq_ignore_ascii_case("Bearer") {
                continue;
            }
            let creds = creds.trim();
            if !creds.is_empty() {
                return Some(creds.to_string());
            }
        } else if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn unauthorized() -> Response {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header("content-type", "application/json")
        // RFC 6750: Bearer challenges must advertise the scheme so
        // standards-compliant clients can negotiate.
        .header("www-authenticate", "Bearer")
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
    fn strips_credential_headers() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("authorization", "Bearer sk-1".parse().unwrap());
        h.insert("api-key", "k2".parse().unwrap());
        h.insert("x-api-key", "k3".parse().unwrap());
        h.insert("user-agent", "curl".parse().unwrap());
        strip_credential_headers(&mut h);
        assert!(h.get("authorization").is_none());
        assert!(h.get("api-key").is_none());
        assert!(h.get("x-api-key").is_none());
        assert_eq!(h.get("user-agent").unwrap(), "curl");
    }

    #[test]
    fn empty_bearer_is_none() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer "));
        assert_eq!(extract_key(&h), None);
    }

    #[test]
    fn bearer_scheme_is_case_insensitive() {
        // RFC 6750: the scheme is case-insensitive.
        let mut h = axum::http::HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("bearer sk-123"));
        assert_eq!(extract_key(&h), Some("sk-123".to_string()));
    }

    #[test]
    fn non_bearer_scheme_is_ignored() {
        // A "Basic ..." Authorization header must not be tried as a raw key
        // and shadow a valid api-key header.
        let mut h = axum::http::HeaderMap::new();
        h.insert(
            "authorization",
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        h.insert("x-api-key", HeaderValue::from_static("sk-123"));
        assert_eq!(extract_key(&h), Some("sk-123".to_string()));

        let mut h = axum::http::HeaderMap::new();
        h.insert(
            "authorization",
            HeaderValue::from_static("Basic dXNlcjpwYXNz"),
        );
        assert_eq!(extract_key(&h), None);
    }

    #[test]
    fn bearer_with_tab_separator_accepted() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer\tsk-123"));
        assert_eq!(extract_key(&h), Some("sk-123".to_string()));
    }

    #[test]
    fn ct_eq_matches_identically_and_rejects_differences() {
        assert!(ct_eq("sk-abcdef", "sk-abcdef"));
        assert!(!ct_eq("sk-abcdef", "sk-abcdeg"));
        assert!(!ct_eq("sk-abc", "sk-abcdef"));
        assert!(!ct_eq("sk-abcdef", "sk-abc"));
        assert!(!ct_eq("", "sk-1"));
        assert!(ct_eq("", ""));
    }

    #[test]
    fn ct_eq_handles_utf8() {
        assert!(ct_eq("ключ-1", "ключ-1"));
        assert!(!ct_eq("ключ-1", "ключ-2"));
    }
}
