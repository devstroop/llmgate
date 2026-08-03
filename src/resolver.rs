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
        for prefix in &self.prefixes {
            if let Some(rest) = name.strip_prefix(prefix.as_str()) {
                return rest;
            }
        }
        name
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
