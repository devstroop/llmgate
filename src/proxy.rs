use std::time::Duration;

/// Forward a JSON body to the upstream conversation endpoint with the given
/// headers (already merged by the caller: authorization + configured extra
/// headers + protocol headers).
pub async fn forward(
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

/// Build the HTTP client used for all upstream traffic. Only a connect
/// timeout is applied — total request time is unbounded so long-lived SSE
/// streams are not cut off.
pub fn client(_timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("failed to build http client")
}
