use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;

use crate::core::privacy::PrivacyConfig;

const DEFAULT_PROTOCOLS: [&str; 2] = ["openai", "anthropic"];
// Loopback by default: an accidentally config-less start must not expose
// an unauthenticated proxy on all interfaces.
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 5000;
const DEFAULT_TIMEOUT_MS: u64 = 60_000;

fn default_host() -> String {
    DEFAULT_HOST.to_string()
}

fn default_port() -> u16 {
    DEFAULT_PORT
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Client-facing protocols to serve. Names must match registered adapters.
    pub client: ClientConfig,
    /// Upstream provider the gateway forwards to.
    pub upstream: UpstreamConfig,
    /// Model resolution settings.
    pub models: ModelsConfig,
    /// HTTP server settings.
    pub server: ServerConfig,
    /// Client request authentication.
    pub auth: AuthConfig,
    /// Reversible redaction (Privacy Guard) settings. Disabled by default.
    #[serde(rename = "privacy_guard")]
    pub privacy: PrivacyConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthConfig {
    /// Accepted client API keys. Empty = all requests allowed. Requests must
    /// present a key via `Authorization: Bearer <key>`, `api-key: <key>` or
    /// `x-api-key: <key>`.
    pub api_keys: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClientConfig {
    pub protocols: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UpstreamConfig {
    /// Protocol name of the upstream adapter (`openai`, `anthropic`, ...).
    pub protocol: String,
    /// Provider base URL, e.g. `https://api.anthropic.com`. The adapter
    /// appends its own conversation path.
    pub url: String,
    /// `Authorization` header value sent upstream. Empty = no header.
    /// For protocols that use a different auth header (e.g. Anthropic's
    /// `x-api-key`), use `extra_headers` instead.
    pub authorization: String,
    /// Additional headers sent on every upstream request, e.g.
    /// `[{name = "x-api-key", value = "..."}]`.
    pub extra_headers: Vec<Header>,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
}

/// An extra header sent on upstream requests. Both fields are REQUIRED:
/// a missing `value` is a configuration error, not an empty header that
/// would be forwarded upstream at request time.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    /// Fallback model when the resolved name is empty.
    pub default: String,
    /// Client-requested model name → upstream model name.
    pub map: HashMap<String, String>,
    /// Prefixes stripped from requested model names before map lookup.
    pub prefixes: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            client: ClientConfig {
                protocols: DEFAULT_PROTOCOLS.iter().map(|s| s.to_string()).collect(),
            },
            upstream: UpstreamConfig {
                protocol: "openai".to_string(),
                url: "http://localhost:11434".to_string(),
                authorization: String::new(),
                extra_headers: Vec::new(),
                timeout_ms: DEFAULT_TIMEOUT_MS,
            },
            models: ModelsConfig {
                default: String::new(),
                map: HashMap::new(),
                prefixes: Vec::new(),
            },
            server: ServerConfig {
                host: DEFAULT_HOST.to_string(),
                port: DEFAULT_PORT,
            },
            auth: AuthConfig {
                api_keys: Vec::new(),
            },
            privacy: PrivacyConfig::default(),
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            protocols: DEFAULT_PROTOCOLS.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            protocol: "openai".to_string(),
            url: "http://localhost:11434".to_string(),
            authorization: String::new(),
            extra_headers: Vec::new(),
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }
}

impl Config {
    /// Load from `CONFIG_PATH` env var or `./config.toml` if present; falls
    /// back to defaults only when neither is set.
    pub fn load() -> anyhow::Result<Self> {
        match std::env::var("CONFIG_PATH") {
            Ok(path) => Self::load_path(&path),
            Err(_) => {
                let default = PathBuf::from("config.toml");
                if default.exists() {
                    Self::load_path(default.to_str().unwrap_or("config.toml"))
                } else {
                    Ok(Self::default())
                }
            }
        }
    }

    /// Load from an explicit path. A missing file is an error: silently
    /// falling back to the defaults would start the gateway without
    /// authentication and pointed at localhost — a typo'd `CONFIG_PATH`
    /// must not produce an open proxy.
    fn load_path(path: &str) -> anyhow::Result<Self> {
        let p = PathBuf::from(path);
        if !p.exists() {
            anyhow::bail!(
                "config file {path:?} does not exist; refusing to fall back to the default (unauthenticated) config"
            );
        }
        let raw = std::fs::read_to_string(&p)?;
        let cfg: Config = toml::from_str(&raw)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Semantic validation that TOML deserialization cannot express.
    fn validate(&self) -> anyhow::Result<()> {
        for key in &self.auth.api_keys {
            // An empty/whitespace API key would make `ct_eq("", "")` accept
            // requests with no credential at all — a silent auth bypass.
            if key.trim().is_empty() {
                anyhow::bail!(
                    "auth.api_keys contains an empty or whitespace-only key; \
                     remove it (an empty key list disables auth explicitly)"
                );
            }
            // The middleware compares against `HeaderValue::to_str()` bytes:
            // a key that cannot be sent as a header or that differs from its
            // trimmed form could never authenticate — a 401-only trap.
            // (`from_str` alone is not enough: the http crate accepts
            // obs-text bytes that `to_str()` still refuses.)
            let invalid_header = match axum::http::HeaderValue::from_str(key) {
                Ok(v) => v.to_str().is_err(),
                Err(_) => true,
            };
            if invalid_header {
                anyhow::bail!(
                    "auth.api_keys contains a key that is not a valid HTTP \
                     header value (visible ASCII required): {key:?}"
                );
            }
            if key != key.trim() {
                anyhow::bail!(
                    "auth.api_keys contains a key with leading/trailing \
                     whitespace ({key:?}); it could never be presented via a \
                     header — trim it"
                );
            }
        }
        // A missing/scheme-less upstream URL fails only at request time;
        // require an absolute http(s) URL up front.
        let parsed = self.upstream.url.parse::<reqwest::Url>().map_err(|e| {
            anyhow::anyhow!(
                "upstream.url {:?} is not a valid absolute URL: {e}",
                self.upstream.url
            )
        })?;
        if !matches!(parsed.scheme(), "http" | "https") {
            anyhow::bail!(
                "upstream.url {:?} must use http or https",
                self.upstream.url
            );
        }
        // Invalid header names would surface as confusing request-time
        // failures; validate at startup.
        for h in &self.upstream.extra_headers {
            if axum::http::HeaderName::from_bytes(h.name.as_bytes()).is_err() {
                anyhow::bail!(
                    "upstream.extra_headers entry {:?} is not a valid HTTP \
                     header name",
                    h.name
                );
            }
        }
        if self.auth.api_keys.is_empty() && !host_is_loopback(&self.server.host) {
            tracing::warn!(
                "authentication is DISABLED (api_keys = []) AND the server \
                 binds {:?} — this gateway is an OPEN PROXY on that \
                 interface. Bind 127.0.0.1 or configure api_keys.",
                self.server.host
            );
        }
        // An empty prefix would match EVERY model name first and defeat
        // the remaining prefixes (and the passthrough default).
        for prefix in &self.models.prefixes {
            if prefix.is_empty() {
                anyhow::bail!("models.prefixes contains an empty string; remove it");
            }
        }
        Ok(())
    }
}

/// True for loopback bind addresses (`127.0.0.0/8`, `::1`, `localhost`).
fn host_is_loopback(host: &str) -> bool {
    host == "localhost" || host == "::1" || host.starts_with("127.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = Config::default();
        assert_eq!(cfg.client.protocols, vec!["openai", "anthropic"]);
        assert_eq!(cfg.server.port, 5000);
    }

    #[test]
    fn validate_rejects_non_header_safe_keys() {
        // A non-ASCII key could never be presented via a header value
        // (to_str() rejects it): the client would get 401s forever.
        let mut cfg = Config::default();
        cfg.auth.api_keys = vec!["sk-\u{00e9}".to_string()];
        assert!(cfg.validate().is_err());
        // A key with leading/trailing whitespace can never match the
        // trimmed presented value.
        cfg.auth.api_keys = vec![" sk-1".to_string()];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_url_and_header_names() {
        let mut cfg = Config::default();
        cfg.upstream.url = "not-a-url".to_string();
        assert!(cfg.validate().is_err(), "scheme-less URL must fail");
        cfg.upstream.url = "file:///etc/passwd".to_string();
        assert!(cfg.validate().is_err(), "non-http(s) URL must fail");
        cfg.upstream.url = "http://127.0.0.1:9090".to_string();
        cfg.upstream.extra_headers = vec![crate::config::Header {
            name: "bad header name!".to_string(),
            value: "v".to_string(),
        }];
        assert!(cfg.validate().is_err(), "invalid header name must fail");
        cfg.upstream.extra_headers = vec![crate::config::Header {
            name: "x-api-key".to_string(),
            value: "v".to_string(),
        }];
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_model_prefix() {
        let mut cfg = Config::default();
        cfg.models.prefixes = vec!["".to_string()];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn no_upstream_url_is_valid_without_check() {
        // TOML serialization enforces url presence; the default config
        // annotated with a URL validates fine (used by other tests).
        let mut cfg = Config::default();
        cfg.upstream.url = "http://127.0.0.1:9090".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn parses_toml() {
        let raw = r#"
[client]
protocols = ["openai"]

[upstream]
protocol = "anthropic"
url = "https://api.anthropic.com"
authorization = "Bearer sk-test"
timeout_ms = 30_000

[models]
default = "claude-3-5-sonnet"
prefixes = ["gateway/"]
[models.map]
"gpt-4o" = "claude-3-5-sonnet"
"#;
        let cfg: Config = toml::from_str(raw).unwrap();
        assert_eq!(cfg.client.protocols, vec!["openai"]);
        assert_eq!(cfg.upstream.protocol, "anthropic");
        assert_eq!(cfg.upstream.timeout_ms, 30_000);
        assert_eq!(cfg.models.default, "claude-3-5-sonnet");
        assert_eq!(cfg.models.map["gpt-4o"], "claude-3-5-sonnet");
        assert_eq!(cfg.server.port, DEFAULT_PORT);
    }

    #[test]
    fn parses_privacy_guard_section() {
        let raw = r#"
[privacy_guard]
enabled = true
vault = "memory"

[[privacy_guard.rules]]
name = "INTERNAL_IP"
pattern = '\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
replacement = "<IP_{n}>"

[privacy_guard.allow_list]
domains = ["devstroop.com"]
patterns = ['\bfoo@example\.com\b']
"#;
        let cfg: Config = toml::from_str(raw).unwrap();
        assert!(cfg.privacy.enabled);
        assert_eq!(cfg.privacy.vault, "memory");
        assert_eq!(cfg.privacy.rules.len(), 1);
        assert_eq!(cfg.privacy.rules[0].name, "INTERNAL_IP");
        assert_eq!(cfg.privacy.rules[0].replacement, "<IP_{n}>");
        assert_eq!(cfg.privacy.allow_list.domains, vec!["devstroop.com"]);
        assert_eq!(cfg.privacy.allow_list.patterns.len(), 1);
    }

    #[test]
    fn privacy_guard_disabled_when_section_absent() {
        let cfg: Config = toml::from_str("[server]\nport = 5001\n").unwrap();
        assert!(!cfg.privacy.enabled);
    }

    #[test]
    fn explicit_config_path_missing_errors() {
        let err = Config::load_path("/nonexistent/model-adapter-test.toml").unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn load_path_reads_existing_file() {
        let dir = std::env::temp_dir().join(format!("ma-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("c.toml");
        std::fs::write(&file, "[server]\nport = 5001\n").unwrap();
        let cfg = Config::load_path(file.to_str().unwrap()).unwrap();
        assert_eq!(cfg.server.port, 5001);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_auth_keys_field_is_rejected() {
        // A typo like `api_key = "..."` (singular) must fail startup
        // instead of silently disabling authentication.
        let raw = "[auth]\napi_key = \"sk-secret\"\n";
        let err = toml::from_str::<Config>(raw).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }

    #[test]
    fn empty_api_key_is_rejected() {
        // `api_keys = [""]` would make ct_eq("", "") accept keyless
        // requests — a silent auth bypass; reject it at startup.
        let raw = "[auth]\napi_keys = [\"\"]\n";
        let cfg: Config = toml::from_str(raw).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");

        let raw = "[auth]\napi_keys = [\"   \"]\n";
        let cfg: Config = toml::from_str(raw).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn extra_header_requires_name_and_value() {
        // A header entry missing `value` must fail deserialization instead
        // of silently becoming an empty header forwarded upstream.
        let raw = "[upstream]\nprotocol = \"openai\"\nextra_headers = [{ name = \"x-api-key\" }]\n";
        let err = toml::from_str::<Config>(raw).unwrap_err();
        assert!(err.to_string().contains("missing field"), "got: {err}");
    }

    #[test]
    fn default_host_is_loopback() {
        // An accidentally config-less start must not expose an
        // unauthenticated proxy on all interfaces.
        assert_eq!(Config::default().server.host, "127.0.0.1");
    }
}
