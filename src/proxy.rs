use std::time::Duration;

/// Forward a JSON body to the upstream conversation endpoint.
pub async fn forward(
    http: &reqwest::Client,
    url: &str,
    protocol_headers: &[(String, String)],
    authorization: &str,
    body: &str,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut req = http
        .post(url)
        .header("content-type", "application/json")
        .body(body.to_string());
    if !authorization.is_empty() {
        req = req.header("authorization", authorization);
    }
    for (key, value) in protocol_headers {
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
