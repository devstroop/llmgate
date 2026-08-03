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

/// Build the HTTP client used for all upstream traffic.
pub fn client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build http client")
}
