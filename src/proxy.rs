use std::time::Duration;

use crate::core::error::AdapterError;

/// Bound on the pre-headers phase of a streaming upstream request. The
/// stream itself stays unbounded; only waiting for the upstream to start
/// responding is bounded, so an upstream that accepts the connection and
/// never sends headers cannot hold the task forever.
const STREAM_HEADER_TIMEOUT: Duration = Duration::from_secs(15);

/// Build the HTTP client used for all upstream traffic.
///
/// Only a connect timeout is set at the client level. Total per-request
/// timeouts are applied individually: non-streaming calls use
/// [`forward`]/[`forward_get`] (via their `timeout` argument), and streaming
/// calls use [`forward_stream`] which has no total timeout so long SSE
/// streams are not cut off (their lifetime is bounded by the protocol:
/// `[DONE]`, `message_stop`, or connection close; the pre-headers phase is
/// bounded by `STREAM_HEADER_TIMEOUT`).
pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        // NEVER follow redirects: reqwest's default policy forwards
        // provider-specific credential headers (x-api-key,
        // x-goog-api-key) to the redirect target — a malicious upstream
        // could exfiltrate them with a 3xx.
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("failed to build http client")
}

/// Forward a JSON body to the upstream conversation endpoint with the given
/// headers. Applies a total per-request `timeout`; use for non-streaming
/// requests only.
pub async fn forward(
    http: &reqwest::Client,
    url: &str,
    headers: &[(String, String)],
    body: &str,
    timeout: Duration,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut req = http.post(url).body(body.to_string()).timeout(timeout);
    for (key, value) in headers {
        req = req.header(key, value);
    }
    // Insert (replace) rather than append so a caller-supplied
    // `content-type` cannot duplicate the body's and get rejected upstream.
    let mut built = req.build()?;
    built.headers_mut().insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    http.execute(built).await
}

/// Forward a streaming request with no total per-request timeout, so long
/// SSE streams are not cut off. The stream ends when the upstream sends
/// `[DONE]`, `message_stop`, or an error, or the connection closes. The
/// pipeline enforces a per-chunk idle timeout to bound stalled upstreams.
pub async fn forward_stream(
    http: &reqwest::Client,
    url: &str,
    headers: &[(String, String)],
    body: &str,
) -> Result<reqwest::Response, AdapterError> {
    // No total timeout on the stream itself (long SSE streams must not be
    // cut off), but the pre-headers phase must not hang forever: an
    // upstream that accepts the connection and never responds would hold
    // the task indefinitely.
    let mut req = http.post(url).body(body.to_string());
    for (key, value) in headers {
        req = req.header(key, value);
    }
    let mut built = req
        .build()
        .map_err(|e| AdapterError::Internal(format!("upstream request build failed: {e}")))?;
    built.headers_mut().insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("application/json"),
    );
    let response = tokio::time::timeout(STREAM_HEADER_TIMEOUT, http.execute(built))
        .await
        .map_err(|_| AdapterError::Upstream {
            status: 504,
            body: format!("upstream did not respond with headers within {STREAM_HEADER_TIMEOUT:?}"),
        })?
        .map_err(|e| AdapterError::Internal(format!("upstream request failed: {e}")))?;
    Ok(response)
}

/// GET an upstream endpoint (used for model listings). Applies a total
/// per-request `timeout`.
pub async fn forward_get(
    http: &reqwest::Client,
    url: &str,
    headers: &[(String, String)],
    timeout: Duration,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut req = http.get(url).timeout(timeout);
    for (key, value) in headers {
        req = req.header(key, value);
    }
    req.send().await
}
