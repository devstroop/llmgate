//! M10 — memory substrate & observability store.
//!
//! An embedded nqlite database (single file + WAL) that records every
//! gateway request as a JSON record: model, protocols, latency, status, and
//! token usage. The engine itself is zero-LLM and deterministic; records are
//! queryable offline with `nql-cli` (lexical `::bm25`, field filters, ...).
//!
//! Design:
//! - A **write-behind actor thread** owns the `nqlite::Database`
//!   (`Database::execute` is single-writer and fsyncs the WAL per mutating
//!   statement). Requests enqueue records via a bounded channel; `record()`
//!   never blocks the request path. A full queue drops the record with a
//!   warning (backpressure policy: observability must not slow the gateway).
//! - **Flush cadence**: the actor checkpoints the WAL into the main file on
//!   an interval (and after draining on shutdown). Until the first
//!   checkpoint the `.nql.wal` is the durable artifact.
//! - **TTL sweep**: when `ttl_hours > 0` the actor tracks inserted record
//!   ids and `FORGET`s expired ones on the same interval (deterministic
//!   cascade removes the record; the ledger keeps memory bounded to live
//!   records).
//! - Fail closed: an enabled `[memory]` section that cannot open its
//!   database aborts startup (see `MemoryStore::start`).

use std::collections::VecDeque;
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nql::parse;
use nqlite::Database;

use crate::config::MemoryConfig;

/// Maximum queued records before the actor drops new ones. Bounded so an
/// observability flood can never grow memory without limit.
const QUEUE_CAP: usize = 4096;
/// Default seconds between WAL checkpoints and TTL sweeps.
const DEFAULT_FLUSH_INTERVAL_SECS: u64 = 30;

/// One gateway request, as recorded in the store.
#[derive(Debug, Clone)]
pub struct RequestRecord {
    /// Inbound `x-request-id`, or "-" when the client sent none.
    pub request_id: String,
    /// Client-facing protocol ("openai", "anthropic", ...).
    pub client: String,
    /// Upstream protocol.
    pub upstream: String,
    /// Resolved model name.
    pub model: String,
    /// "stream" or "non-stream".
    pub mode: &'static str,
    /// "ok" or "error".
    pub status: &'static str,
    /// Wall time from request start to completion.
    pub latency_ms: u64,
    /// Token usage from the response; `None` when the upstream reported none
    /// (streams are recorded without usage in M10 — see PLAN.md).
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

/// Handle to the write-behind memory actor.
pub struct MemoryStore {
    tx: SyncSender<RequestRecord>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MemoryStore {
    /// Open (or create) the nqlite database at `cfg.path`, declare the
    /// `request` table (idempotent in the engine), and spawn the actor.
    /// Fails closed: any error here aborts gateway startup.
    pub fn start(cfg: &MemoryConfig) -> anyhow::Result<Self> {
        let path = cfg
            .path
            .as_deref()
            .filter(|p| !p.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("memory.path is required when memory.enabled = true"))?;

        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    anyhow::anyhow!("memory: cannot create directory for {path:?}: {e}")
                })?;
            }
        }

        let mut db = Database::open(path)
            .map_err(|e| anyhow::anyhow!("memory: failed to open nqlite database {path:?}: {e}"))?;
        // CREATE TABLE is a no-op re-declaration in the engine, so this is
        // safe on an existing database.
        let plan = parse("CREATE TABLE request;")
            .map_err(|e| anyhow::anyhow!("memory: internal error building CREATE TABLE: {e}"))?;
        db.execute(&plan)
            .map_err(|e| anyhow::anyhow!("memory: failed to initialize request table: {e}"))?;

        let (tx, rx) = mpsc::sync_channel::<RequestRecord>(QUEUE_CAP);
        let flush_interval = Duration::from_secs(
            cfg.flush_interval_secs
                .unwrap_or(DEFAULT_FLUSH_INTERVAL_SECS)
                .max(1),
        );
        let ttl_ms = cfg.ttl_hours.unwrap_or(0).saturating_mul(3_600_000);

        let handle = thread::Builder::new()
            .name("llmgate-memory".into())
            .spawn(move || actor(rx, db, flush_interval, ttl_ms))
            .map_err(|e| anyhow::anyhow!("memory: failed to spawn writer thread: {e}"))?;

        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    /// Fire-and-forget: enqueue a record. Never blocks; on a full queue the
    /// record is dropped with a warning (bounded, non-blocking by design).
    pub fn record(&self, rec: RequestRecord) {
        if let Err(e) = self.tx.try_send(rec) {
            tracing::warn!(
                ?e,
                "memory: record queue full; dropping observability record"
            );
        }
    }

    /// Drop the sender and wait for the actor to drain + checkpoint.
    /// Used by tests and graceful shutdown; idempotent.
    pub fn shutdown(&mut self) {
        self.tx = mpsc::sync_channel(0).0;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Unix milliseconds, as stored in record bodies.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Serialize one record into its `INSERT INTO request:<id> {...}` statement.
/// `id` and `ts_ms` are generated ONCE by the caller so the TTL ledger and
/// the stored body always agree.
fn insert_statement(id: &str, ts_ms: u64, rec: &RequestRecord) -> Result<String, String> {
    let body = serde_json::json!({
        "ts_ms": ts_ms,
        "request_id": rec.request_id,
        "client": rec.client,
        "upstream": rec.upstream,
        "model": rec.model,
        "mode": rec.mode,
        "status": rec.status,
        "latency_ms": rec.latency_ms,
        "input_tokens": rec.input_tokens,
        "output_tokens": rec.output_tokens,
    })
    .to_string();
    Ok(format!("INSERT INTO request:{id} {body};"))
}

/// FORGET records older than `cutoff_ms`; returns the number forgotten.
fn sweep(db: &mut Database, ledger: &mut VecDeque<(String, u64)>, cutoff_ms: u64) -> usize {
    let mut expired = Vec::new();
    while let Some((_, ts)) = ledger.front() {
        if *ts < cutoff_ms {
            if let Some((id, _)) = ledger.pop_front() {
                expired.push(id);
            }
        } else {
            break;
        }
    }
    if expired.is_empty() {
        return 0;
    }
    let plan_text = expired
        .iter()
        .map(|id| format!("FORGET request:{id};"))
        .collect::<String>();
    match parse(&plan_text) {
        Ok(plan) => match db.execute(&plan) {
            Ok(_) => expired.len(),
            Err(e) => {
                tracing::warn!(%e, "memory: TTL sweep failed");
                0
            }
        },
        Err(e) => {
            tracing::warn!(%e, "memory: TTL sweep plan failed");
            0
        }
    }
}

fn insert_one(db: &mut Database, seq: &mut u64, rec: &RequestRecord) -> Option<(String, u64)> {
    *seq += 1;
    let ts = now_ms();
    let id = format!("r{ts}_{seq}");
    match insert_statement(&id, ts, rec) {
        Ok(text) => match parse(&text) {
            Ok(plan) => match db.execute(&plan) {
                Ok(_) => Some((id, ts)),
                Err(e) => {
                    tracing::warn!(%e, "memory: insert failed");
                    None
                }
            },
            Err(e) => {
                tracing::warn!(%e, "memory: insert plan failed");
                None
            }
        },
        Err(e) => {
            tracing::warn!(%e, "memory: statement build failed");
            None
        }
    }
}

/// Writer loop: execute queued inserts, checkpoint + sweep on the flush
/// interval, and drain + final checkpoint when all senders drop.
fn actor(
    rx: mpsc::Receiver<RequestRecord>,
    mut db: Database,
    flush_interval: Duration,
    ttl_ms: u64,
) {
    let mut seq: u64 = 0;
    let mut dirty = false;
    // (record id, ts_ms) ledger for TTL sweeps; only tracked when a TTL is set.
    let track = ttl_ms > 0;
    let mut ledger: VecDeque<(String, u64)> = VecDeque::new();

    loop {
        match rx.recv_timeout(flush_interval) {
            Ok(rec) => {
                if let Some((id, ts)) = insert_one(&mut db, &mut seq, &rec) {
                    dirty = true;
                    if track {
                        ledger.push_back((id, ts));
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if dirty {
                    if let Err(e) = db.flush() {
                        tracing::warn!(%e, "memory: checkpoint failed");
                    }
                    dirty = false;
                }
                if track && !ledger.is_empty() {
                    let cutoff = now_ms().saturating_sub(ttl_ms);
                    sweep(&mut db, &mut ledger, cutoff);
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    // Shutdown drain: process anything left, then one final checkpoint.
    while let Ok(rec) = rx.try_recv() {
        if insert_one(&mut db, &mut seq, &rec).is_some() {
            dirty = true;
        }
    }
    if dirty {
        let _ = db.flush();
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cfg(dir: &std::path::Path) -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            path: Some(dir.join("mem.nql").to_string_lossy().into_owned()),
            ttl_hours: Some(0),
            flush_interval_secs: Some(1),
        }
    }

    fn rec(n: u64) -> RequestRecord {
        RequestRecord {
            request_id: format!("req-{n}"),
            client: "openai".into(),
            upstream: "openai".into(),
            model: "gpt-4o".into(),
            mode: "non-stream",
            status: "ok",
            latency_ms: 12 + n,
            input_tokens: Some(10 + n),
            output_tokens: Some(20 + n),
        }
    }

    #[test]
    fn actor_persists_records_and_reopens() {
        let dir = std::env::temp_dir().join(format!("llmgate-mem-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let cfg = temp_cfg(&dir);
        let mut store = MemoryStore::start(&cfg).unwrap();
        store.record(rec(1));
        store.record(rec(2));
        store.shutdown();

        let mut db = Database::open(cfg.path.as_deref().unwrap()).unwrap();
        let plan = parse("SELECT * FROM request ORDER BY ::recency;").unwrap();
        let results = db.execute(&plan).unwrap();
        let rows = &results[0].rows;
        assert_eq!(rows.len(), 2, "both records must survive shutdown+reopen");
        // nql-ir `Value` serializes as a tagged enum, so the stored body is
        // {"field": {"Str": ...} / {"Num": ...}} — unwrap the tag.
        let body0 = serde_json::to_value(&rows[0].record.body).unwrap();
        assert_eq!(body0["status"]["Str"], "ok");
        assert_eq!(body0["model"]["Str"], "gpt-4o");
        assert_eq!(body0["request_id"]["Str"], "req-1");
        assert_eq!(body0["input_tokens"]["Int"].as_i64(), Some(11));
        assert_eq!(body0["mode"]["Str"], "non-stream");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sweep_forgets_expired_records() {
        let dir = std::env::temp_dir().join(format!("llmgate-mem-sweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let cfg = temp_cfg(&dir);
        // Starting the store declares the request table; then we open the
        // same file directly to drive the sweep against a known record id.
        let mut store = MemoryStore::start(&cfg).unwrap();
        store.shutdown();

        let mut db = Database::open(cfg.path.as_deref().unwrap()).unwrap();
        // Insert one record under a KNOWN id so the sweep can target it, and
        // seed a ledger that references it as expired.
        let plan = parse(r#"INSERT INTO request:r999_1 { "ts_ms": 1, "status": "ok" };"#).unwrap();
        db.execute(&plan).unwrap();
        let mut ledger: VecDeque<(String, u64)> = VecDeque::new();
        ledger.push_back(("r999_1".into(), 1_000));
        let forgotten = sweep(&mut db, &mut ledger, now_ms() - 60_000);
        assert_eq!(forgotten, 1);
        assert!(ledger.is_empty());

        let plan = parse("SELECT * FROM request;").unwrap();
        let results = db.execute(&plan).unwrap();
        assert_eq!(results[0].rows.len(), 0, "FORGET must cascade the record");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn insert_statement_shape() {
        let text = insert_statement("r1_7", 42, &rec(1)).unwrap();
        assert!(text.starts_with("INSERT INTO request:r1_7"), "{text}");
        assert!(text.contains("\"ts_ms\":42"), "{text}");
        assert!(text.contains("\"model\":\"gpt-4o\""), "{text}");
        assert!(text.contains("\"status\":\"ok\""), "{text}");
        assert!(text.ends_with("};"), "{text}");
        // The statement must parse.
        assert!(parse(&text).is_ok());
    }
}
