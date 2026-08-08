use std::collections::HashMap;

/// Protocol-agnostic model-name resolution pipeline:
/// strip prefix → model_map lookup → default.
#[derive(Debug, Clone)]
pub struct ModelResolver {
    default: String,
    map: HashMap<String, String>,
    prefixes: Vec<String>,
}

impl ModelResolver {
    pub fn new(default: String, map: HashMap<String, String>, prefixes: Vec<String>) -> Self {
        Self {
            default,
            map,
            prefixes,
        }
    }

    pub fn resolve(&self, requested: &str) -> String {
        let stripped = self.strip_prefix(requested);
        if let Some(mapped) = self.map.get(stripped) {
            return mapped.clone();
        }
        if stripped.is_empty() {
            return self.default.clone();
        }
        stripped.to_string()
    }

    fn strip_prefix<'a>(&self, name: &'a str) -> &'a str {
        // Longest-prefix match: `["vendor/", "vendor/gpt/"]` must strip
        // `vendor/gpt/` from `vendor/gpt/4o` regardless of declaration
        // order (a short prefix listed first would silently win).
        let mut best: Option<(usize, &'a str)> = None;
        for prefix in &self.prefixes {
            if let Some(rest) = name.strip_prefix(prefix.as_str()) {
                if best.is_none_or(|(best_len, _)| prefix.len() > best_len) {
                    best = Some((prefix.len(), rest));
                }
            }
        }
        best.map(|(_, rest)| rest).unwrap_or(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver() -> ModelResolver {
        let mut map = HashMap::new();
        map.insert("gpt-4o".to_string(), "claude-3-5-sonnet".to_string());
        ModelResolver::new(
            "fallback-model".to_string(),
            map,
            vec!["vendor/".to_string(), "gateway/".to_string()],
        )
    }

    #[test]
    fn longest_prefix_wins_regardless_of_order() {
        // `vendor/gpt/4o` must strip `vendor/gpt/` -> `4o` (longest
        // prefix), never the shorter `vendor/` -> `gpt/4o`.
        for prefixes in [
            vec!["vendor/".to_string(), "vendor/gpt/".to_string()],
            vec!["vendor/gpt/".to_string(), "vendor/".to_string()],
        ] {
            let r = ModelResolver::new("default".into(), HashMap::new(), prefixes);
            assert_eq!(r.resolve("vendor/gpt/4o"), "4o");
        }
    }

    #[test]
    fn passthrough_unknown_model() {
        assert_eq!(resolver().resolve("some-model"), "some-model");
    }

    #[test]
    fn strips_prefix_before_map() {
        assert_eq!(resolver().resolve("vendor/gpt-4o"), "claude-3-5-sonnet");
    }

    #[test]
    fn maps_without_prefix() {
        assert_eq!(resolver().resolve("gpt-4o"), "claude-3-5-sonnet");
    }

    #[test]
    fn strips_prefix_passthrough() {
        assert_eq!(resolver().resolve("gateway/deepseek"), "deepseek");
    }

    #[test]
    fn empty_uses_default() {
        assert_eq!(resolver().resolve("vendor/"), "fallback-model");
        assert_eq!(resolver().resolve(""), "fallback-model");
    }
}
