use std::time::Duration;

/// Build the HTTP client used for all upstream traffic.
///
/// Only a connect timeout is set at the client level. Total per-request
/// timeouts are applied individually: non-streaming calls use
/// [`forward`]/[`forward_get`] (via their `timeout` argument), and streaming
/// calls use [`forward_stream`] which has no total timeout so long SSE
/// streams are not cut off (their lifetime is bounded by the protocol:
/// `[DONE]`, `message_stop`, or connection close).
pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
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
    let mut req = http
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_string())
        .timeout(timeout);
    for (key, value) in headers {
        req = req.header(key, value);
    }
    req.send().await
}

/// Forward a streaming request with no total per-request timeout, so long
/// SSE streams are not cut off. The stream ends when the upstream sends
/// `[DONE]`, `message_stop`, or an error, or the connection closes.
pub async fn forward_stream(
    http: &reqwest::Client,
    url: &str,
    headers: &[(String, String)],
    body: &str,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut req = http
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_string());
    for (key, value) in headers {
        req = req.header(key, value);
    }
    req.send().await
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
