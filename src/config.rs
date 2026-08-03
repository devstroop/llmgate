use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;

const DEFAULT_PROTOCOLS: [&str; 2] = ["openai", "anthropic"];
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: u16 = 5000;
const DEFAULT_TIMEOUT_MS: u64 = 60_000;

fn default_host() -> String {
    DEFAULT_HOST.to_string()
}

fn default_port() -> u16 {
    DEFAULT_PORT
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Client-facing protocols to serve. Names must match registered adapters.
    pub client: ClientConfig,
    /// Upstream provider the gateway forwards to.
    pub upstream: UpstreamConfig,
    /// Model resolution settings.
    pub models: ModelsConfig,
    /// HTTP server settings.
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ClientConfig {
    pub protocols: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct UpstreamConfig {
    /// Protocol name of the upstream adapter (`openai`, `anthropic`, ...).
    pub protocol: String,
    /// Provider base URL, e.g. `https://api.anthropic.com`. The adapter
    /// appends its own conversation path.
    pub url: String,
    /// `Authorization` header value sent upstream. Empty = no header.
    pub authorization: String,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// Fallback model when the resolved name is empty.
    pub default: String,
    /// Client-requested model name → upstream model name.
    pub map: HashMap<String, String>,
    /// Prefixes stripped from requested model names before map lookup.
    pub prefixes: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
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
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }
}

impl Config {
    /// Load from `CONFIG_PATH` env var or `./config.toml` if present; falls
    /// back to defaults when neither exists.
    pub fn load() -> anyhow::Result<Self> {
        let path = std::env::var("CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("config.toml"));
        if path.exists() {
            let raw = std::fs::read_to_string(&path)?;
            Ok(toml::from_str(&raw)?)
        } else {
            Ok(Self::default())
        }
    }
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
}
