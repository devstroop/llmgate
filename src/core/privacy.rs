//! Reversible redaction (pseudonymization) for the neutral pipeline.
//!
//! Privacy Guard is a session-scoped, reversible tokenizer. Sensitive
//! entities in the inbound `NeutralRequest` (PII, API keys, IPs, custom
//! rule matches) are replaced with synthetic tokens (`<EMAIL_1>`,
//! `<IP_2>`, ...) before the request is serialized for the upstream
//! provider. The token → original mapping lives in a session-scoped
//! vault for the lifetime of the request. As the upstream response
//! streams back, tokens are restored to the original values before the
//! client sees them — the client experiences zero degradation, the
//! upstream provider sees zero sensitive data.
//!
//! # Token grammar
//!
//! Tokens use the form `<NAME_n>` — a `>` terminator makes boundaries
//! unambiguous: `<IP_1>` can never match inside `<IP_10>`, and no token
//! is a prefix of another token, so a leftmost Aho-Corasick match over a
//! sliding buffer is a safe streaming replacement strategy.
//!
//! # Out of scope (v1)
//!
//! - Image blocks are never redacted (no OCR/vision pass).
//! - Tool *definitions* (`NeutralTool`) are not scanned — only message
//!   content blocks and tool-call input JSON.
//! - The vault is in-memory and session-scoped by design: mappings die
//!   with the request. A persistent vault (e.g. sqlite) is a future
//!   backend behind the same session API.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use aho_corasick::{AhoCorasick, MatchKind};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

use super::error::AdapterError;
use super::neutral::{
    ContentBlock, NeutralMessage, NeutralRequest, NeutralResponse, NeutralStreamEvent,
};

/// Cap on tokens per session. Beyond this, redaction stops rather than
/// growing the vault unboundedly (guards against token-blasting).
pub const MAX_SESSION_TOKENS: usize = 4096;

/// Conservative built-in rule set, used when the feature is enabled with
/// no custom rules configured. Order matters: rules apply sequentially.
const DEFAULT_RULES: &[(&str, &str, &str)] = &[
    (
        "EMAIL",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "<EMAIL_{n}>",
    ),
    (
        "IPV4",
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
        "<IP_{n}>",
    ),
    ("PHONE", r"\b\d{3}[-.)]\d{3}[-.]\d{4}\b", "<PHONE_{n}>"),
    (
        "API_KEY",
        r"\b(?:sk|pk)-[A-Za-z0-9]{20,64}\b",
        "<SECRET_{n}>",
    ),
];

/// `[privacy_guard]` configuration. Disabled by default.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrivacyConfig {
    /// Master switch. When `false` (default) the pipeline is untouched.
    pub enabled: bool,
    /// Vault backend. Only `"memory"` (session-scoped, in-process) is
    /// implemented; anything else is a configuration error so the feature
    /// fails closed rather than silently running unredacted.
    pub vault: String,
    /// Custom redaction rules. When `enabled` and `rules` is empty, the
    /// built-in conservative set (email, IPv4, US phone, API keys) is
    /// used.
    pub rules: Vec<RuleConfig>,
    /// Matches that must never be redacted.
    pub allow_list: AllowListConfig,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            vault: "memory".to_string(),
            rules: Vec::new(),
            allow_list: AllowListConfig::default(),
        }
    }
}

/// One custom redaction rule.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleConfig {
    /// Rule name; also the label inside generated tokens.
    pub name: String,
    /// Regex matched against each text block.
    pub pattern: String,
    /// Replacement template; must contain `{n}`, substituted with a
    /// per-session counter (e.g. `<IP_{n}>` → `<IP_1>`, `<IP_2>`, ...).
    pub replacement: String,
}

/// Matches that are exempt from redaction.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AllowListConfig {
    /// Domains whose emails/URLs are never redacted (exact or subdomain),
    /// e.g. `["devstroop.com"]` allows `alice@devstroop.com`.
    pub domains: Vec<String>,
    /// Regexes: a match matching any of these is never redacted.
    pub patterns: Vec<String>,
}

#[derive(Debug)]
struct CompiledRule {
    name: String,
    regex: Regex,
    template: String,
}

/// Compiled, immutable redaction engine, built once at startup and shared
/// by every request session.
#[derive(Debug)]
pub struct RedactionEngine {
    rules: Vec<CompiledRule>,
    allow_domains: Vec<String>,
    allow_patterns: Vec<Regex>,
    /// Longest token this engine can mint (template length + counter
    /// slack). Bounds the streaming restorer's hold-back tail.
    max_token_len: usize,
}

/// Can this pattern produce a zero-length match on realistic text?
///
/// `is_match("")` is not enough: a word-boundary assertion (`\b`) does not
/// match the empty haystack yet yields zero-length matches inside text, and
/// `x*`-style patterns match empty at most positions. Probe strings cover
/// word/non-word/unicode boundaries; any zero-length match rejects the
/// rule (empty matches would mint tokens for "" and corrupt the text).
fn can_match_empty(regex: &Regex) -> bool {
    for probe in ["a", "a b", " a", "a ", "123", "é", " "] {
        if regex.find_iter(probe).any(|m| m.start() == m.end()) {
            return true;
        }
    }
    regex.is_match("")
}

/// Enforce the `<NAME_{n}>` token grammar for a custom rule's replacement.
fn validate_template(rule: &RuleConfig) -> anyhow::Result<()> {
    let t = &rule.replacement;
    let bad = |what: &str| {
        anyhow::anyhow!(
            "privacy_guard.rules[{}]: replacement {what} (token grammar is <NAME_{{n}}>, e.g. <CRED_{{n}}>)",
            rule.name
        )
    };
    let Some(rest) = t.strip_prefix('<') else {
        return Err(bad("must start with a single '<'"));
    };
    let Some(rest) = rest.strip_suffix('>') else {
        return Err(bad("must end with a single '>'"));
    };
    if rest.matches('{').count() != 1 || rest.matches('}').count() != 1 || !rest.ends_with("{n}") {
        return Err(bad(
            "must contain exactly one {{n}} slot immediately before the closing '>'",
        ));
    }
    let name = &rest[..rest.len() - 3]; // strip "{n}"
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(bad(
            "name must be non-empty and use only [A-Za-z0-9_] (no '<', '>', spaces, or punctuation)",
        ));
    }
    Ok(())
}

/// The template's token name: everything between the leading '<' and the
/// trailing `{n}>` (4 characters).
fn template_prefix(template: &str) -> &str {
    &template[1..template.len() - 4]
}

/// Can two token names produce the same token for some counter pair?
///
/// Names that differ only by a digit suffix overlap: `<X_` at counter 11
/// mints `<X_11>` and `<X_1` at counter 1 mints `<X_11>` too. (A suffix
/// with a leading zero cannot collide — counters never render with leading
/// zeros — and identical names are handled by the duplicate check.)
fn digit_suffix_collides(pa: &str, pb: &str) -> bool {
    let (long, short) = if pa.len() >= pb.len() {
        (pa, pb)
    } else {
        (pb, pa)
    };
    if !long.starts_with(short) {
        return false;
    }
    let suffix = &long[short.len()..];
    !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) && !suffix.starts_with('0')
}

impl RedactionEngine {
    /// Compile a `[privacy_guard]` config. Fails closed: any invalid rule
    /// pattern, template, or vault backend is an error, not a warning.
    pub fn new(config: &PrivacyConfig) -> anyhow::Result<Arc<Self>> {
        if config.vault != "memory" {
            anyhow::bail!(
                "privacy_guard.vault: unsupported backend {:?} (only \"memory\" is implemented)",
                config.vault
            );
        }

        let rules = if config.rules.is_empty() {
            DEFAULT_RULES
                .iter()
                .map(|(name, pattern, replacement)| CompiledRule {
                    name: (*name).to_string(),
                    regex: Regex::new(pattern).expect("built-in rule compiles"),
                    template: (*replacement).to_string(),
                })
                .collect()
        } else {
            let mut rules = Vec::with_capacity(config.rules.len());
            for rule in &config.rules {
                if rule.name.trim().is_empty() {
                    anyhow::bail!("privacy_guard.rules: rule name must not be empty");
                }
                // Token grammar `<NAME_{n}>`, strictly enforced: exactly
                // one leading '<', a plain `[A-Za-z0-9_]+` name, exactly
                // one `{n}` slot, and a single trailing '>'. The streaming
                // hold-back assumes the LAST '<' in the buffer is the only
                // possible token start, so '<'/'>' must not appear
                // anywhere else, and the closing '>' guarantees no token is
                // a prefix of another (`<X_1>` can never match inside
                // `<X_10>`).
                validate_template(rule)?;
                let regex = Regex::new(&rule.pattern).map_err(|e| {
                    anyhow::anyhow!("privacy_guard.rules[{}]: invalid pattern: {e}", rule.name)
                })?;
                if can_match_empty(&regex) {
                    anyhow::bail!(
                        "privacy_guard.rules[{}]: pattern can match the empty string (would mint empty tokens and corrupt text); require at least one character",
                        rule.name
                    );
                }
                rules.push(CompiledRule {
                    name: rule.name.clone(),
                    regex,
                    template: rule.replacement.clone(),
                });
            }
            rules
        };

        // Every rule needs its own token namespace: identical templates
        // collide outright, and templates whose names differ only by a
        // digit suffix can collide for some counters (`<X_{n}>` at counter
        // 11 mints `<X_11>`, which equals `<X_1{n}>` at counter 1).
        for (i, a) in rules.iter().enumerate() {
            for b in &rules[i + 1..] {
                let (pa, pb) = (template_prefix(&a.template), template_prefix(&b.template));
                if pa == pb {
                    anyhow::bail!(
                        "privacy_guard.rules: rules {:?} and {:?} use the same replacement template {:?}; every rule needs its own token namespace",
                        a.name,
                        b.name,
                        a.template
                    );
                }
                if digit_suffix_collides(pa, pb) {
                    anyhow::bail!(
                        "privacy_guard.rules: rules {:?} and {:?} have colliding replacement templates ({:?} vs {:?}): counters overlap; pick non-overlapping names",
                        a.name,
                        b.name,
                        a.template,
                        b.template
                    );
                }
            }
        }

        let allow_patterns = config
            .allow_list
            .patterns
            .iter()
            .map(|p| {
                Regex::new(p).map_err(|e| {
                    anyhow::anyhow!("privacy_guard.allow_list.patterns: invalid pattern: {e}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Longest possible token: longest template minus the `{n}` slot
        // plus room for the counter digits.
        let max_token_len = rules
            .iter()
            .map(|r| r.template.len().saturating_sub(3) + 12)
            .max()
            .unwrap_or(16)
            .max(16);

        Ok(Arc::new(Self {
            rules,
            allow_domains: config.allow_list.domains.clone(),
            allow_patterns,
            max_token_len,
        }))
    }

    pub fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    /// Is this match exempt (allow-list domain or pattern)?
    fn is_allowed(&self, matched: &str) -> bool {
        if self.allow_patterns.iter().any(|re| re.is_match(matched)) {
            return true;
        }
        if self.allow_domains.is_empty() {
            return false;
        }
        // Candidate domain: for emails the part after the last '@', for
        // URLs the host after "://", otherwise the match itself.
        let candidate: &str = if matched.contains('@') {
            matched.rsplit('@').next().unwrap_or(matched)
        } else if let Some((_, rest)) = matched.split_once("://") {
            rest.split(['/', '?', '#']).next().unwrap_or(rest)
        } else {
            matched
        };
        let candidate = candidate.trim();
        self.allow_domains.iter().any(|d| {
            let d = d.trim();
            !d.is_empty() && (candidate == d || candidate.ends_with(&format!(".{d}")))
        })
    }
}

#[derive(Default)]
struct Vault {
    /// token → original value
    token_to_original: HashMap<String, String>,
    /// (rule name, original value) → token; repeated values reuse a token
    /// so the model sees a consistent symbol for one entity.
    value_to_token: HashMap<(String, String), String>,
    /// per-rule counters
    counters: HashMap<String, u64>,
    /// bumped on every new token; invalidates the automaton cache
    version: u64,
}

/// One request's redaction session: the token vault plus restore
/// machinery. Created per request after parsing, dropped when the response
/// has been fully restored — mappings never outlive the request.
pub struct RedactionSession {
    engine: Arc<RedactionEngine>,
    vault: Mutex<Vault>,
    /// Cached Aho-Corasick automaton over the vault's tokens, rebuilt when
    /// the vault version changes (new tokens minted since last build).
    automaton: RwLock<Option<(u64, AhoCorasick)>>,
}

impl RedactionSession {
    pub fn new(engine: Arc<RedactionEngine>) -> Self {
        Self {
            engine,
            vault: Mutex::new(Vault::default()),
            automaton: RwLock::new(None),
        }
    }

    pub fn engine(&self) -> &RedactionEngine {
        &self.engine
    }

    /// Number of tokens minted so far (for telemetry).
    pub fn token_count(&self) -> usize {
        self.vault.lock().unwrap().token_to_original.len()
    }

    /// Redact a request in place. Applies every compiled rule to every
    /// text-bearing content block. Image blocks are intentionally left
    /// untouched (v1 scope). Fails closed when the session token cap is
    /// hit: the caller must reject the request rather than forward
    /// unredacted data upstream. The input is only replaced on success, so
    /// a failed call never leaves a partially redacted request behind.
    ///
    /// MEMORY TRADEOFF: redaction works on a clone so a failure (e.g. cap
    /// exhaustion) leaves the caller's request untouched. That doubles peak
    /// memory for privacy-enabled requests (up to 2x the 16 MiB inbound
    /// cap). The alternative — in-place mutation with a rollback log — is
    /// error-prone under partial failure; operators enabling the guard on
    /// memory-constrained hosts should account for ~2x request size.
    pub fn redact_request(&self, req: &mut NeutralRequest) -> Result<(), AdapterError> {
        let mut work = req.clone();
        for message in &mut work.messages {
            self.redact_message(message)?;
        }
        *req = work;
        Ok(())
    }

    fn redact_message(&self, message: &mut NeutralMessage) -> Result<(), AdapterError> {
        for block in &mut message.content {
            match block {
                ContentBlock::Text(t) => *t = self.redact_text(t)?,
                ContentBlock::Thinking { thinking, .. } => {
                    *thinking = self.redact_text(thinking)?
                }
                ContentBlock::ToolResult { content, .. } => *content = self.redact_text(content)?,
                ContentBlock::ToolUse { input, .. } => self.redact_json(input)?,
                ContentBlock::Image { .. } | ContentBlock::RedactedThinking { .. } => {}
            }
        }
        Ok(())
    }

    /// Replace all rule matches in `text` with tokens, respecting the
    /// allow-list and the per-session token cap (fail closed on cap
    /// exhaustion).
    pub fn redact_text(&self, text: &str) -> Result<String, AdapterError> {
        let mut out = text.to_string();
        for rule in &self.engine.rules {
            let mut result = String::with_capacity(out.len());
            let mut last = 0;
            let mut changed = false;
            // Tokens minted by earlier rules must never be re-scanned by a
            // later rule (a broad pattern could match inside `<EMAIL_1>`
            // and corrupt the token so restore cannot recover it).
            let token_spans: Vec<(usize, usize)> = self
                .with_automaton(|ac| ac.find_iter(&out).map(|m| (m.start(), m.end())).collect())
                .unwrap_or_default();
            for caps in rule.regex.captures_iter(&out) {
                let m = caps.get(0).expect("regex group 0 exists");
                if token_spans
                    .iter()
                    .any(|&(s, e)| m.start() < e && m.end() > s)
                {
                    continue;
                }
                result.push_str(&out[last..m.start()]);
                if self.engine.is_allowed(m.as_str()) {
                    result.push_str(m.as_str());
                } else {
                    result.push_str(&self.token_for(rule, m.as_str())?);
                }
                last = m.end();
                changed = true;
            }
            if changed {
                result.push_str(&out[last..]);
                out = result;
            }
        }
        Ok(out)
    }

    /// Mint (or reuse) the token for `original` under `rule`.
    fn token_for(&self, rule: &CompiledRule, original: &str) -> Result<String, AdapterError> {
        let mut vault = self.vault.lock().unwrap();
        let key = (rule.name.clone(), original.to_string());
        if let Some(token) = vault.value_to_token.get(&key) {
            return Ok(token.clone());
        }
        if vault.token_to_original.len() >= MAX_SESSION_TOKENS {
            // Fail closed: never forward the remaining matches unredacted.
            // The request must be rejected by the caller.
            return Err(AdapterError::Internal(format!(
                "privacy guard: session token cap ({MAX_SESSION_TOKENS}) reached; refusing to forward unredacted data"
            )));
        }
        let n = vault.counters.entry(rule.name.clone()).or_insert(0);
        *n += 1;
        let token = rule.template.replace("{n}", &n.to_string());
        // Fail closed against cross-rule token collisions (e.g. templates
        // whose counters can overlap): never overwrite an existing mapping
        // with a different original, and never forward the sensitive value
        // unredacted — the caller must reject the request.
        if let Some(existing) = vault.token_to_original.get(&token) {
            if existing != original {
                return Err(AdapterError::Internal(format!(
                    "privacy guard: token {token} already maps to a different original (colliding rule templates?); refusing to forward unredacted data"
                )));
            }
        }
        vault
            .token_to_original
            .insert(token.clone(), original.to_string());
        vault.value_to_token.insert(key, token.clone());
        vault.version += 1;
        Ok(token)
    }

    /// Look up the original value for a token.
    pub fn lookup(&self, token: &str) -> Option<String> {
        self.vault
            .lock()
            .unwrap()
            .token_to_original
            .get(token)
            .cloned()
    }

    /// Restore a complete text in one pass (non-streaming responses).
    pub fn restore_text(&self, text: &str) -> String {
        self.with_automaton(|ac| {
            let mut out = String::with_capacity(text.len());
            let mut last = 0;
            for m in ac.find_iter(text) {
                let token = &text[m.start()..m.end()];
                out.push_str(&text[last..m.start()]);
                out.push_str(&self.lookup(token).unwrap_or_else(|| token.to_string()));
                last = m.end();
            }
            out.push_str(&text[last..]);
            out
        })
        .unwrap_or_else(|| text.to_string())
    }

    /// Restore every text-bearing block of a parsed upstream response.
    pub fn restore_response(&self, resp: &mut NeutralResponse) {
        for block in &mut resp.content {
            match block {
                ContentBlock::Text(t) => *t = self.restore_text(t),
                ContentBlock::Thinking { thinking, .. } => *thinking = self.restore_text(thinking),
                ContentBlock::ToolUse { input, .. } => self.restore_json(input),
                ContentBlock::ToolResult { content, .. } => *content = self.restore_text(content),
                ContentBlock::Image { .. } | ContentBlock::RedactedThinking { .. } => {}
            }
        }
    }

    fn redact_json(&self, value: &mut Value) -> Result<(), AdapterError> {
        match value {
            Value::String(s) => *s = self.redact_text(s)?,
            Value::Array(items) => {
                for item in items {
                    self.redact_json(item)?;
                }
            }
            Value::Object(map) => {
                for v in map.values_mut() {
                    self.redact_json(v)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn restore_json(&self, value: &mut Value) {
        match value {
            Value::String(s) => *s = self.restore_text(s),
            Value::Array(items) => {
                for item in items {
                    self.restore_json(item);
                }
            }
            Value::Object(map) => {
                for v in map.values_mut() {
                    self.restore_json(v);
                }
            }
            _ => {}
        }
    }

    /// Run `f` against an automaton covering the vault's current tokens,
    /// rebuilding it if the vault has grown since the last build. Returns
    /// `None` when the vault holds no tokens (restore is a no-op).
    fn with_automaton<T>(&self, f: impl FnOnce(&AhoCorasick) -> T) -> Option<T> {
        let version = self.vault.lock().unwrap().version;
        {
            let cached = self.automaton.read().unwrap();
            if let Some((v, ac)) = cached.as_ref() {
                if *v == version {
                    return Some(f(ac));
                }
            }
        }
        let mut cached = self.automaton.write().unwrap();
        let version = self.vault.lock().unwrap().version;
        if let Some((v, ac)) = cached.as_ref() {
            if *v == version {
                return Some(f(ac));
            }
        }
        let patterns: Vec<String> = {
            let vault = self.vault.lock().unwrap();
            vault.token_to_original.keys().cloned().collect()
        };
        if patterns.is_empty() {
            return None;
        }
        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostFirst)
            .build(&patterns)
            .expect("valid token patterns");
        let result = f(&ac);
        *cached = Some((version, ac));
        Some(result)
    }
}

/// Streaming token restorer. Buffers fragments so tokens split across
/// upstream SSE chunks are restored correctly; plain text is emitted with
/// bounded latency (at most one token length behind).
pub struct RestoreStream {
    session: Arc<RedactionSession>,
    buffer: String,
    max_token_len: usize,
    /// When true, restored originals are JSON-escaped on splice. Used for
    /// the tool-arguments channel, which carries raw JSON text: an original
    /// containing `"`, `\`, or control characters must be escaped so the
    /// delta stream stays valid JSON (mirrors the non-streaming path, which
    /// escapes via `Value` serialization).
    json_escape: bool,
}

impl RestoreStream {
    pub fn new(session: Arc<RedactionSession>) -> Self {
        Self::with_mode(session, false)
    }

    /// A restorer for a channel carrying raw JSON text (tool arguments).
    pub fn new_json(session: Arc<RedactionSession>) -> Self {
        Self::with_mode(session, true)
    }

    fn with_mode(session: Arc<RedactionSession>, json_escape: bool) -> Self {
        Self {
            max_token_len: session.engine().max_token_len(),
            session,
            buffer: String::new(),
            json_escape,
        }
    }

    /// Feed one fragment; returns the restorable text emitted so far (may
    /// be empty while a token is still incomplete).
    pub fn feed(&mut self, fragment: &str) -> String {
        self.buffer.push_str(fragment);
        let mut out = String::new();
        loop {
            let matched = self
                .session
                .with_automaton(|ac| ac.find(&self.buffer).map(|m| (m.start(), m.end())))
                .flatten();
            let Some((start, end)) = matched else {
                break;
            };
            out.push_str(&self.buffer[..start]);
            let token = &self.buffer[start..end];
            let original = self
                .session
                .lookup(token)
                .unwrap_or_else(|| token.to_string());
            if self.json_escape {
                // serde_json::to_string yields a quoted, escaped JSON
                // string; strip the surrounding quotes.
                let quoted = serde_json::to_string(&original).expect("string serializes");
                out.push_str(&quoted[1..quoted.len() - 1]);
            } else {
                out.push_str(&original);
            }
            self.buffer.drain(..end);
        }
        // Bound the hold-back: only text that could still become a token
        // is retained. A token must start with '<' and fit within
        // `max_token_len` bytes, so any '<' before the last
        // `max_token_len - 1` bytes would have completed inside the
        // buffer (and been matched above) — everything before it is safe
        // to emit, which keeps latency bounded on plain-text runs.
        // `window_start` is a raw byte offset: round it down to a char
        // boundary so slicing multi-byte UTF-8 text cannot panic.
        let mut window_start = self.buffer.len().saturating_sub(self.max_token_len - 1);
        while window_start > 0 && !self.buffer.is_char_boundary(window_start) {
            window_start -= 1;
        }
        let keep = self.buffer[window_start..]
            .find('<')
            .map(|i| window_start + i)
            .unwrap_or(self.buffer.len());
        if keep > 0 {
            out.push_str(&self.buffer[..keep]);
            self.buffer.drain(..keep);
        }
        out
    }

    /// Flush any held tail (the stream ended before a token completed;
    /// emit it literally).
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }
}

/// Per-stream restore state: one [`RestoreStream`] per text channel so
/// tokens cannot bleed between text, reasoning, and tool-argument streams.
pub struct StreamRestorer {
    session: Arc<RedactionSession>,
    text: RestoreStream,
    reasoning: RestoreStream,
    tool_args: HashMap<u32, RestoreStream>,
}

impl StreamRestorer {
    pub fn new(session: Arc<RedactionSession>) -> Self {
        Self {
            session: session.clone(),
            text: RestoreStream::new(session.clone()),
            reasoning: RestoreStream::new(session.clone()),
            tool_args: HashMap::new(),
        }
    }

    /// Restore one neutral event. Returns `None` when there is nothing to
    /// emit (an empty restored delta).
    pub fn restore_event(&mut self, event: NeutralStreamEvent) -> Option<NeutralStreamEvent> {
        match event {
            NeutralStreamEvent::TextDelta(t) => {
                let restored = self.text.feed(&t);
                if restored.is_empty() {
                    None
                } else {
                    Some(NeutralStreamEvent::TextDelta(restored))
                }
            }
            NeutralStreamEvent::ReasoningDelta(t) => {
                let restored = self.reasoning.feed(&t);
                if restored.is_empty() {
                    None
                } else {
                    Some(NeutralStreamEvent::ReasoningDelta(restored))
                }
            }
            NeutralStreamEvent::ToolCallDelta {
                index,
                id,
                name,
                arguments,
            } => {
                let restored = self
                    .tool_args
                    .entry(index)
                    .or_insert_with(|| RestoreStream::new_json(self.session.clone()))
                    .feed(&arguments);
                // Emit nothing only when the delta carries no information
                // at all (an incomplete token being held back, with no new
                // call metadata). A delta that opens a new call (id/name)
                // is meaningful even with empty arguments.
                if restored.is_empty() && id.is_empty() && name.is_empty() {
                    None
                } else {
                    Some(NeutralStreamEvent::ToolCallDelta {
                        index,
                        id,
                        name,
                        arguments: restored,
                    })
                }
            }
            NeutralStreamEvent::Error(e) => {
                // An upstream SSE error chunk may echo the redacted
                // request (validation errors quote the offending input);
                // restore any minted tokens before the client sees them.
                let msg = e.to_string();
                let restored = self.session.restore_text(&msg);
                Some(NeutralStreamEvent::Error(AdapterError::Api(restored)))
            }
            other => Some(other),
        }
    }

    /// Flush partial tokens held across all channels (stream ended).
    pub fn finish(&mut self) -> Vec<NeutralStreamEvent> {
        let mut events = Vec::new();
        let text = self.text.finish();
        if !text.is_empty() {
            events.push(NeutralStreamEvent::TextDelta(text));
        }
        let reasoning = self.reasoning.finish();
        if !reasoning.is_empty() {
            events.push(NeutralStreamEvent::ReasoningDelta(reasoning));
        }
        let mut args: Vec<(u32, String)> = self
            .tool_args
            .drain()
            .map(|(index, mut rs)| (index, rs.finish()))
            .filter(|(_, s)| !s.is_empty())
            .collect();
        args.sort_by_key(|(index, _)| *index);
        for (index, arguments) in args {
            events.push(NeutralStreamEvent::ToolCallDelta {
                index,
                id: String::new(),
                name: String::new(),
                arguments,
            });
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_with(config: PrivacyConfig) -> Arc<RedactionSession> {
        let engine = RedactionEngine::new(&config).expect("engine compiles");
        Arc::new(RedactionSession::new(engine))
    }

    fn default_session() -> Arc<RedactionSession> {
        session_with(PrivacyConfig {
            enabled: true,
            ..PrivacyConfig::default()
        })
    }

    fn default_engine() -> Arc<RedactionEngine> {
        RedactionEngine::new(&PrivacyConfig {
            enabled: true,
            ..PrivacyConfig::default()
        })
        .expect("engine compiles")
    }

    #[test]
    fn disabled_by_default() {
        let config = PrivacyConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.vault, "memory");
        assert!(config.rules.is_empty());
    }

    #[test]
    fn enabled_without_rules_uses_builtin_defaults() {
        let engine = default_engine();
        assert!(engine.rules.len() >= 4, "expected built-in default rules");
        let session = Arc::new(RedactionSession::new(engine));
        let redacted = session
            .redact_text("write to john.doe@example.com or 10.1.2.3")
            .unwrap();
        assert_eq!(redacted, "write to <EMAIL_1> or <IP_1>");
        assert_eq!(session.token_count(), 2);
    }

    #[test]
    fn redacts_email_and_ipv4() {
        let session = default_session();
        let redacted = session
            .redact_text("Contact john.doe@example.com or use 192.168.1.50.")
            .unwrap();
        assert_eq!(redacted, "Contact <EMAIL_1> or use <IP_1>.");
        assert_eq!(
            session.lookup("<EMAIL_1>").as_deref(),
            Some("john.doe@example.com")
        );
        assert_eq!(session.lookup("<IP_1>").as_deref(), Some("192.168.1.50"));
    }

    #[test]
    fn reuses_token_for_repeated_value() {
        let session = default_session();
        let redacted = session
            .redact_text("IP 192.168.1.50 and again 192.168.1.50")
            .unwrap();
        assert_eq!(redacted, "IP <IP_1> and again <IP_1>");
        assert_eq!(session.token_count(), 1);
    }

    #[test]
    fn allow_list_domain_skips_email() {
        let session = session_with(PrivacyConfig {
            enabled: true,
            allow_list: AllowListConfig {
                domains: vec!["devstroop.com".to_string()],
                ..AllowListConfig::default()
            },
            ..PrivacyConfig::default()
        });
        let redacted = session
            .redact_text("mail alice@devstroop.com but also bob@example.com")
            .unwrap();
        assert_eq!(redacted, "mail alice@devstroop.com but also <EMAIL_1>");
    }

    #[test]
    fn allow_list_domain_covers_subdomains() {
        let session = session_with(PrivacyConfig {
            enabled: true,
            allow_list: AllowListConfig {
                domains: vec!["devstroop.com".to_string()],
                ..AllowListConfig::default()
            },
            ..PrivacyConfig::default()
        });
        let redacted = session.redact_text("reach dev@api.devstroop.com").unwrap();
        assert_eq!(redacted, "reach dev@api.devstroop.com");
    }

    #[test]
    fn allow_list_pattern_skips_match() {
        let session = session_with(PrivacyConfig {
            enabled: true,
            allow_list: AllowListConfig {
                patterns: vec![r"^192\.168\.".to_string()],
                ..AllowListConfig::default()
            },
            ..PrivacyConfig::default()
        });
        let redacted = session
            .redact_text("lan 192.168.1.50, wan 8.8.8.8")
            .unwrap();
        assert_eq!(redacted, "lan 192.168.1.50, wan <IP_1>");
    }

    #[test]
    fn custom_rule_replacement_template() {
        let session = session_with(PrivacyConfig {
            enabled: true,
            rules: vec![RuleConfig {
                name: "PROJECT".to_string(),
                pattern: r"\b[A-Z]{2,}-\d{3}\b".to_string(),
                replacement: "<PROJECT_{n}>".to_string(),
            }],
            ..PrivacyConfig::default()
        });
        let redacted = session.redact_text("deploy ACME-123 and ACME-999").unwrap();
        assert_eq!(redacted, "deploy <PROJECT_1> and <PROJECT_2>");
        assert_eq!(session.lookup("<PROJECT_2>").as_deref(), Some("ACME-999"));
        assert_eq!(
            session.restore_text(&redacted),
            "deploy ACME-123 and ACME-999"
        );
    }

    #[test]
    fn non_stream_round_trip() {
        let session = default_session();
        let original = "Contact john.doe@example.com, server 192.168.1.50, phone 555-123-4567.";
        let redacted = session.redact_text(original).unwrap();
        assert!(!redacted.contains("john.doe@example.com"));
        assert!(!redacted.contains("192.168.1.50"));
        assert_eq!(session.restore_text(&redacted), original);
    }

    #[test]
    fn restore_leaves_unknown_tokens_untouched() {
        let session = default_session();
        session.redact_text("use 192.168.1.50").unwrap();
        let restored = session.restore_text("<IP_1> vs <IP_99>");
        assert_eq!(restored, "192.168.1.50 vs <IP_99>");
    }

    #[test]
    fn token_terminator_prevents_prefix_collision() {
        // `<IP_1>` must never match inside `<IP_10>`: the `>` terminator
        // makes token boundaries unambiguous.
        let session = default_session();
        let mut ips = String::new();
        for i in 1..=10 {
            ips.push_str(&format!("10.0.0.{i} "));
        }
        let redacted = session.redact_text(&ips).unwrap();
        assert!(redacted.contains("<IP_10>"));
        assert_eq!(
            session.restore_text("IPs <IP_1> and <IP_10>"),
            "IPs 10.0.0.1 and 10.0.0.10"
        );
        assert_eq!(session.restore_text("<IP_10>"), "10.0.0.10");
    }

    #[test]
    fn fragmented_stream_restore() {
        let session = default_session();
        session.redact_text("use 192.168.1.50 now").unwrap();
        let mut stream = RestoreStream::new(session);
        let mut out = String::new();
        for fragment in ["<", "IP", "_1", ">", " reached"] {
            out.push_str(&stream.feed(fragment));
        }
        out.push_str(&stream.finish());
        assert_eq!(out, "192.168.1.50 reached");
    }

    #[test]
    fn stream_restore_multiple_tokens_across_fragments() {
        let session = default_session();
        session
            .redact_text("192.168.1.50 and john@example.com")
            .unwrap();
        let mut stream = RestoreStream::new(session);
        let mut out = String::new();
        for fragment in ["The IP is <IP", "_1> and email <EMAIL", "_1> ok"] {
            out.push_str(&stream.feed(fragment));
        }
        out.push_str(&stream.finish());
        assert_eq!(out, "The IP is 192.168.1.50 and email john@example.com ok");
    }

    #[test]
    fn stream_restore_large_plain_runs_stay_bounded() {
        let session = default_session();
        session.redact_text("secret 192.168.1.50").unwrap();
        let mut stream = RestoreStream::new(session);
        let big = "a".repeat(10_000);
        let mut out = String::new();
        for fragment in big.as_bytes().chunks(333) {
            out.push_str(&stream.feed(std::str::from_utf8(fragment).unwrap()));
        }
        out.push_str(&stream.feed("<IP_1>"));
        out.push_str(&stream.finish());
        assert_eq!(out.len(), big.len() + "192.168.1.50".len());
        assert_eq!(&out[..10_000], big);
        assert!(out.ends_with("192.168.1.50"));
        assert!(stream.buffer.len() <= stream.max_token_len);
    }

    #[test]
    fn stream_finish_flushes_incomplete_token_literally() {
        let session = default_session();
        session.redact_text("x 192.168.1.50").unwrap();
        let mut stream = RestoreStream::new(session);
        assert_eq!(stream.feed("prefix <IP"), "prefix ");
        assert_eq!(stream.finish(), "<IP");
    }

    #[test]
    fn stream_no_false_positive_on_plain_text() {
        let session = default_session();
        session.redact_text("192.168.1.50").unwrap();
        let mut stream = RestoreStream::new(session);
        let out = stream.feed("no tokens here <EMAIL_99> or <IP_99>") + &stream.finish();
        assert_eq!(out, "no tokens here <EMAIL_99> or <IP_99>");
    }

    #[test]
    fn request_blocks_redacted_and_round_trip() {
        let session = default_session();
        let mut req = NeutralRequest::new(
            "m",
            vec![
                NeutralMessage {
                    role: super::super::neutral::NeutralRole::User,
                    content: vec![
                        ContentBlock::Text("my ip is 192.168.1.50".to_string()),
                        ContentBlock::Image {
                            media_type: "image/png".to_string(),
                            base64: "iVBORw0KGgo=".to_string(),
                        },
                    ],
                },
                NeutralMessage {
                    role: super::super::neutral::NeutralRole::Assistant,
                    content: vec![ContentBlock::Thinking {
                        thinking: "recall john@example.com".to_string(),
                        signature: None,
                    }],
                },
            ],
        );
        session.redact_request(&mut req).unwrap();
        assert_eq!(
            req.messages[0].content[0],
            ContentBlock::Text("my ip is <IP_1>".into())
        );
        // Images are out of scope: untouched.
        assert!(matches!(
            &req.messages[0].content[1],
            ContentBlock::Image { base64, .. } if base64 == "iVBORw0KGgo="
        ));
        assert!(matches!(
            &req.messages[1].content[0],
            ContentBlock::Thinking { thinking, .. } if thinking == "recall <EMAIL_1>"
        ));

        let mut resp = NeutralResponse {
            id: "r1".to_string(),
            model: "m".to_string(),
            content: vec![ContentBlock::Text(
                "blocked <IP_1> for <EMAIL_1>".to_string(),
            )],
            finish_reason: super::super::neutral::FinishReason::Stop,
            usage: None,
        };
        session.restore_response(&mut resp);
        assert_eq!(
            resp.content[0],
            ContentBlock::Text("blocked 192.168.1.50 for john@example.com".to_string())
        );
    }

    #[test]
    fn tool_input_json_redacted_and_restored() {
        let session = default_session();
        let mut req = NeutralRequest::new(
            "m",
            vec![NeutralMessage {
                role: super::super::neutral::NeutralRole::User,
                content: vec![ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "lookup".to_string(),
                    input: serde_json::json!({
                        "email": "alice@example.com",
                        "nested": { "note": "ip 192.168.1.50" },
                        "count": 3,
                    }),
                }],
            }],
        );
        session.redact_request(&mut req).unwrap();
        let ContentBlock::ToolUse { input, .. } = &req.messages[0].content[0] else {
            panic!("expected tool use");
        };
        assert_eq!(input["email"], "<EMAIL_1>");
        assert_eq!(input["nested"]["note"], "ip <IP_1>");
        assert_eq!(input["count"], 3);

        let mut resp = NeutralResponse {
            id: "r1".to_string(),
            model: "m".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "lookup".to_string(),
                input: serde_json::json!({ "email": "<EMAIL_1>" }),
            }],
            finish_reason: super::super::neutral::FinishReason::Stop,
            usage: None,
        };
        session.restore_response(&mut resp);
        let ContentBlock::ToolUse { input, .. } = &resp.content[0] else {
            panic!("expected tool use");
        };
        assert_eq!(input["email"], "alice@example.com");
    }

    #[test]
    fn stream_restorer_restores_tokens_in_error_events() {
        // An upstream SSE error chunk may echo the redacted request; the
        // client must see its own values, not the minted tokens.
        let session = default_session();
        session.redact_text("mail 192.168.1.50").unwrap();
        let mut restorer = StreamRestorer::new(session);
        let restored = restorer.restore_event(NeutralStreamEvent::Error(AdapterError::Api(
            "invalid <IP_1>".to_string(),
        )));
        match restored {
            Some(NeutralStreamEvent::Error(e)) => {
                assert!(e.to_string().contains("192.168.1.50"));
                assert!(!e.to_string().contains("<IP_1>"));
            }
            other => panic!("expected Error event, got {other:?}"),
        }
    }

    #[test]
    fn stream_restorer_restores_held_token_at_finish() {
        let session = default_session();
        session
            .redact_request(&mut NeutralRequest::new(
                "m",
                vec![NeutralMessage {
                    role: super::super::neutral::NeutralRole::User,
                    content: vec![ContentBlock::Text("192.168.1.50".to_string())],
                }],
            ))
            .unwrap();
        let mut restorer = StreamRestorer::new(session);

        assert_eq!(
            restorer.restore_event(NeutralStreamEvent::MessageStart {
                id: "r1".into(),
                model: "m".into(),
                usage: None,
            }),
            Some(NeutralStreamEvent::MessageStart {
                id: "r1".into(),
                model: "m".into(),
                usage: None,
            })
        );
        // Token split across two deltas, both on the text channel.
        assert_eq!(
            restorer.restore_event(NeutralStreamEvent::TextDelta("<IP".into())),
            None
        );
        assert_eq!(
            restorer.restore_event(NeutralStreamEvent::TextDelta("_1> ok".into())),
            Some(NeutralStreamEvent::TextDelta("192.168.1.50 ok".into()))
        );
        // Reasoning channel has its own buffer.
        assert_eq!(
            restorer.restore_event(NeutralStreamEvent::ReasoningDelta("nope <IP".into())),
            Some(NeutralStreamEvent::ReasoningDelta("nope ".into()))
        );
        assert_eq!(
            restorer.restore_event(NeutralStreamEvent::ReasoningDelta("_1>".into())),
            Some(NeutralStreamEvent::ReasoningDelta("192.168.1.50".into()))
        );
        // Empty deltas are dropped.
        assert_eq!(
            restorer.restore_event(NeutralStreamEvent::TextDelta(String::new())),
            None
        );

        let flushed = restorer.finish();
        assert!(flushed.is_empty(), "no partial tokens left: {flushed:?}");
    }

    #[test]
    fn stream_restorer_tool_args_channel() {
        let session = default_session();
        session.redact_text("192.168.1.50").unwrap();
        let mut restorer = StreamRestorer::new(session);
        let first = restorer.restore_event(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
            arguments: "{\"ip\":\"<IP".into(),
        });
        // The plain head streams through; the incomplete token is held.
        assert!(
            matches!(first, Some(NeutralStreamEvent::ToolCallDelta { arguments, .. }) if arguments == "{\"ip\":\"")
        );
        let second = restorer.restore_event(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: String::new(),
            name: String::new(),
            arguments: "_1>\"}".into(),
        });
        assert!(
            matches!(second, Some(NeutralStreamEvent::ToolCallDelta { arguments, .. }) if arguments == "192.168.1.50\"}")
        );
    }

    #[test]
    fn invalid_vault_backend_fails_closed() {
        let err = RedactionEngine::new(&PrivacyConfig {
            enabled: true,
            vault: "nqlite".to_string(),
            ..PrivacyConfig::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("unsupported backend"));
    }

    #[test]
    fn invalid_replacement_template_rejected() {
        // Templates that violate the strict `<NAME_{n}>` grammar would
        // break the streaming restorer's hold-back assumptions.
        for (replacement, needle) in [
            ("FOO", "must start with a single '<'"),
            ("<F<O{n}>", "must be non-empty and use only"),
            ("<FOO{n}{n}>", "must contain exactly one"),
            ("<FOO>", "must contain exactly one"),
            ("<F O_{n}>", "must be non-empty and use only"),
            ("<<X_{n}>", "must be non-empty and use only"),
        ] {
            let err = RedactionEngine::new(&PrivacyConfig {
                enabled: true,
                rules: vec![RuleConfig {
                    name: "X".to_string(),
                    pattern: r"\bfoo\b".to_string(),
                    replacement: replacement.to_string(),
                }],
                ..PrivacyConfig::default()
            })
            .unwrap_err();
            assert!(
                err.to_string().contains(needle),
                "template {replacement:?}: expected {needle:?}, got {err}"
            );
        }
    }

    #[test]
    fn digit_suffix_templates_rejected() {
        // `<X_{n}>` at counter 11 mints `<X_11>` == `<X_1{n}>` at counter 1:
        // such pairs must fail at engine build time.
        let err = RedactionEngine::new(&PrivacyConfig {
            enabled: true,
            rules: vec![
                RuleConfig {
                    name: "A".to_string(),
                    pattern: r"\bfoo\b".to_string(),
                    replacement: "<X_{n}>".to_string(),
                },
                RuleConfig {
                    name: "B".to_string(),
                    pattern: r"\bbar\b".to_string(),
                    replacement: "<X_1{n}>".to_string(),
                },
            ],
            ..PrivacyConfig::default()
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("colliding replacement templates"),
            "got: {err}"
        );
    }

    #[test]
    fn invalid_rule_pattern_rejected() {
        let err = RedactionEngine::new(&PrivacyConfig {
            enabled: true,
            rules: vec![RuleConfig {
                name: "X".to_string(),
                pattern: "([unclosed".to_string(),
                replacement: "<X_{n}>".to_string(),
            }],
            ..PrivacyConfig::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("invalid pattern"));
    }

    #[test]
    fn session_token_cap_fails_closed() {
        let session = default_session();
        let mut text = String::new();
        for i in 0..(MAX_SESSION_TOKENS + 8) {
            text.push_str(&format!("user{i}@example.com "));
        }
        // Exhausting the cap must REJECT the request — the remaining
        // matches must never be forwarded unredacted.
        let err = session.redact_text(&text).unwrap_err();
        assert!(err.to_string().contains("cap"));
    }

    #[test]
    fn empty_matching_pattern_rejected() {
        // Patterns that can match the empty string would mint empty tokens
        // and corrupt text; reject them at engine build time.
        for bad in [r"\b", r"x*", r"[A-Za-z]*"] {
            let err = RedactionEngine::new(&PrivacyConfig {
                enabled: true,
                rules: vec![RuleConfig {
                    name: "X".into(),
                    pattern: bad.into(),
                    replacement: "<X_{n}>".into(),
                }],
                ..PrivacyConfig::default()
            })
            .unwrap_err();
            assert!(
                err.to_string().contains("empty string"),
                "unexpected error for {bad:?}: {err}"
            );
        }
    }

    #[test]
    fn later_rules_do_not_rematch_minted_tokens() {
        // A broad later rule (e.g. \d+) must not match inside tokens
        // minted by an earlier rule, or restore cannot recover the
        // original.
        let session = session_with(PrivacyConfig {
            enabled: true,
            rules: vec![
                RuleConfig {
                    name: "EMAIL".into(),
                    pattern: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b".into(),
                    replacement: "<EMAIL_{n}>".into(),
                },
                RuleConfig {
                    name: "DIGITS".into(),
                    pattern: r"\d+".into(),
                    replacement: "<NUM_{n}>".into(),
                },
            ],
            ..PrivacyConfig::default()
        });
        let redacted = session.redact_text("mail a@b.com, id 12345").unwrap();
        // The '1' inside <EMAIL_1> must survive the DIGITS rule untouched.
        assert_eq!(redacted, "mail <EMAIL_1>, id <NUM_1>");
        assert_eq!(session.restore_text(&redacted), "mail a@b.com, id 12345");
    }

    #[test]
    fn stream_restore_multibyte_text_does_not_panic() {
        // Regression: the hold-back window was computed from raw byte
        // offsets and could land inside a multi-byte UTF-8 char, panicking
        // on `buffer[window_start..]`. CJK text > max_token_len exercises it.
        let session = default_session();
        session.redact_text("secret 192.168.1.50").unwrap();
        let mut stream = RestoreStream::new(session);
        let big = "日本語テキストです。".repeat(40);
        let chars: Vec<char> = big.chars().collect();
        let mut out = String::new();
        for chunk in chars.chunks(9) {
            out.push_str(&stream.feed(&chunk.iter().collect::<String>()));
        }
        out.push_str(&stream.feed("<IP_1>"));
        out.push_str(&stream.finish());
        assert_eq!(out.len(), big.len() + "192.168.1.50".len());
        assert!(out.ends_with("192.168.1.50"));
    }

    #[test]
    fn replacement_without_token_grammar_rejected() {
        // Tokens must be `<NAME_{n}>`-shaped: the streaming hold-back
        // assumes a '<' start, and the counter must be closed by '>' so no
        // token is a prefix of another (TOK1 vs TOK10).
        for bad in ["TOK{n}", "<TOK{n}", "<TOK{n}X>", "{n}>"] {
            let err = RedactionEngine::new(&PrivacyConfig {
                enabled: true,
                rules: vec![RuleConfig {
                    name: "X".into(),
                    pattern: r"\bfoo\b".into(),
                    replacement: bad.into(),
                }],
                ..PrivacyConfig::default()
            })
            .unwrap_err();
            assert!(
                err.to_string().contains("must start with a single '<'")
                    || err.to_string().contains("must end with a single '>'")
                    || err.to_string().contains("must contain exactly one"),
                "unexpected error for {bad:?}: {err}"
            );
        }
    }

    #[test]
    fn duplicate_replacement_template_across_rules_rejected() {
        // Two rules minting the same token namespace would overwrite each
        // other's vault entries and restore the wrong original.
        let err = RedactionEngine::new(&PrivacyConfig {
            enabled: true,
            rules: vec![
                RuleConfig {
                    name: "A".into(),
                    pattern: r"\ba\b".into(),
                    replacement: "<X_{n}>".into(),
                },
                RuleConfig {
                    name: "B".into(),
                    pattern: r"\bb\b".into(),
                    replacement: "<X_{n}>".into(),
                },
            ],
            ..PrivacyConfig::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("same replacement template"));
    }

    #[test]
    fn tool_args_restore_json_escapes_originals() {
        // The tool-args channel carries raw JSON text; an original
        // containing JSON metacharacters must be escaped on splice so the
        // restored delta stays valid JSON (the non-streaming path escapes
        // via Value serialization — streaming must match).
        let session = session_with(PrivacyConfig {
            enabled: true,
            rules: vec![RuleConfig {
                name: "QUOTED".into(),
                pattern: r#""[^"]*""#.into(),
                replacement: "<QUOTED_{n}>".into(),
            }],
            ..PrivacyConfig::default()
        });
        session.redact_text("say \"hello world\" now").unwrap();
        let mut restorer = StreamRestorer::new(session);
        let first = restorer.restore_event(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: "call_1".into(),
            name: "f".into(),
            arguments: "{\"msg\":\"<QUOTED".into(),
        });
        let second = restorer.restore_event(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: String::new(),
            name: String::new(),
            arguments: "_1>\"}".into(),
        });
        let mut args = String::new();
        if let Some(NeutralStreamEvent::ToolCallDelta { arguments, .. }) = first {
            args.push_str(&arguments);
        }
        if let Some(NeutralStreamEvent::ToolCallDelta { arguments, .. }) = second {
            args.push_str(&arguments);
        }
        let v: Value =
            serde_json::from_str(&args).expect("restored tool args must stay valid JSON");
        assert_eq!(v["msg"], "\"hello world\"");
    }

    #[test]
    fn stream_restorer_drops_empty_tool_call_delta() {
        let session = default_session();
        session.redact_text("192.168.1.50").unwrap();
        let mut restorer = StreamRestorer::new(session);
        // A delta that only holds back an incomplete token (no content, no
        // id/name) has nothing to emit.
        let held = restorer.restore_event(NeutralStreamEvent::ToolCallDelta {
            index: 0,
            id: String::new(),
            name: String::new(),
            arguments: "<IP".into(),
        });
        assert!(held.is_none());
        // A delta that starts a new call (id/name) must still be emitted
        // even with empty arguments.
        let started = restorer.restore_event(NeutralStreamEvent::ToolCallDelta {
            index: 1,
            id: "call_2".into(),
            name: "f".into(),
            arguments: String::new(),
        });
        assert!(started.is_some());
    }

    #[test]
    fn multiple_rules_apply_in_order() {
        let session = session_with(PrivacyConfig {
            enabled: true,
            rules: vec![
                RuleConfig {
                    name: "CRED".to_string(),
                    pattern: r"\b[A-Za-z0-9]{8}\b".to_string(),
                    replacement: "<CRED_{n}>".to_string(),
                },
                RuleConfig {
                    name: "EMAIL".to_string(),
                    pattern: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b".to_string(),
                    replacement: "<EMAIL_{n}>".to_string(),
                },
            ],
            ..PrivacyConfig::default()
        });
        let redacted = session
            .redact_text("password hunter22, mail a@b.com")
            .unwrap();
        // "password" and "hunter22" (8 chars each) are redacted by the
        // first rule; the email by the second.
        assert_eq!(redacted, "<CRED_1> <CRED_2>, mail <EMAIL_1>");
        assert_eq!(
            session.restore_text(&redacted),
            "password hunter22, mail a@b.com"
        );
    }
}
