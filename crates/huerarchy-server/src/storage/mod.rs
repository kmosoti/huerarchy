pub mod error;
pub mod migrations;
pub mod paths;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use huerarchy_core::{
    Capability, CapabilityId, EvidenceKind, EvidenceRef, Observation, ObservationFilter,
    ObservationId, ObservationPage, ObservationStatement, ParticipantId,
};
use rusqlite::{params, Connection, OptionalExtension};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use error::StorageError;
pub use migrations::CURRENT_SCHEMA_VERSION;
pub use paths::{ensure_database_dir, resolve_database_path, HUERARCHY_DB_PATH_ENV};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PublishStatus {
    Created,
    AlreadyExists,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PublishResult {
    pub status: PublishStatus,
    pub observation: Observation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct StorageHealth {
    pub path: String,
    pub journal_mode: String,
    pub schema_version: i32,
    pub observation_count: usize,
    pub capability_count: usize,
}

/// Thread-safe SQLite persistence handle backed by WAL mode.
#[derive(Clone)]
pub struct Storage {
    conn: Arc<Mutex<Connection>>,
    db_path: PathBuf,
}

impl Storage {
    /// Opens or creates a SQLite database at the specified path.
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        ensure_database_dir(path)?;
        let mut conn = Connection::open(path)?;
        configure_connection(&mut conn, false)?;
        migrations::run_migrations(&mut conn)?;

        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path: path.to_path_buf(),
        };
        storage.seed_built_in_capabilities()?;
        Ok(storage)
    }

    /// Opens an in-memory SQLite database, primarily for testing.
    pub fn open_in_memory() -> Result<Self, StorageError> {
        let mut conn = Connection::open_in_memory()?;
        configure_connection(&mut conn, true)?;
        migrations::run_migrations(&mut conn)?;

        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path: PathBuf::from(":memory:"),
        };
        storage.seed_built_in_capabilities()?;
        Ok(storage)
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Health check and statistics for the storage engine.
    pub fn health(&self) -> Result<StorageHealth, StorageError> {
        let conn = self.conn.lock().unwrap();
        let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0))?;
        let schema_version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        let observation_count: usize =
            conn.query_row("SELECT COUNT(*) FROM observations", [], |r| {
                r.get::<_, i64>(0).map(|c| c as usize)
            })?;
        let capability_count: usize =
            conn.query_row("SELECT COUNT(*) FROM capabilities", [], |r| {
                r.get::<_, i64>(0).map(|c| c as usize)
            })?;

        Ok(StorageHealth {
            path: self.db_path.to_string_lossy().to_string(),
            journal_mode,
            schema_version,
            observation_count,
            capability_count,
        })
    }

    /// Registers a capability contract.
    pub fn register_capability(&self, cap: &Capability) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        let schema_json = serde_json::to_string(&cap.contract_schema)?;
        let created_at_rfc3339 = cap.created_at.to_rfc3339();

        conn.execute(
            r#"
            INSERT INTO capabilities (id, name, description, version, contract_schema, provider, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                version = excluded.version,
                contract_schema = excluded.contract_schema,
                provider = excluded.provider
            "#,
            params![
                cap.id.as_str(),
                &cap.name,
                &cap.description,
                &cap.version,
                &schema_json,
                cap.provider.as_str(),
                &created_at_rfc3339,
            ],
        )?;

        Ok(())
    }

    /// Lists registered capabilities in deterministic order.
    pub fn list_capabilities(&self) -> Result<Vec<Capability>, StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, description, version, contract_schema, provider, created_at
             FROM capabilities
             ORDER BY id ASC",
        )?;

        let rows = stmt.query_map([], |row| {
            let id_str: String = row.get(0)?;
            let name: String = row.get(1)?;
            let description: String = row.get(2)?;
            let version: String = row.get(3)?;
            let schema_str: String = row.get(4)?;
            let provider_str: String = row.get(5)?;
            let created_at_str: String = row.get(6)?;

            Ok((
                id_str,
                name,
                description,
                version,
                schema_str,
                provider_str,
                created_at_str,
            ))
        })?;

        let mut caps = Vec::new();
        for r in rows {
            let (id_str, name, description, version, schema_str, provider_str, created_at_str) = r?;
            let id = CapabilityId::new(id_str)?;
            let provider = ParticipantId::new(provider_str)?;
            let contract_schema: serde_json::Value = serde_json::from_str(&schema_str)?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|e| StorageError::Migration(e.to_string()))?
                .with_timezone(&Utc);

            caps.push(Capability::new(
                id,
                name,
                description,
                version,
                contract_schema,
                provider,
                created_at,
            )?);
        }

        Ok(caps)
    }

    /// Records an observation transactionally with an explicit idempotency policy.
    pub fn publish_observation(&self, obs: Observation) -> Result<PublishResult, StorageError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Idempotency check: see if observation with same ID exists
        if let Some(existing) = load_observation_by_id(&tx, obs.id.as_str())? {
            if existing.author == obs.author
                && existing.subject == obs.subject
                && existing.statement == obs.statement
                && existing.evidence == obs.evidence
            {
                // Identical contents: idempotent replay
                return Ok(PublishResult {
                    status: PublishStatus::AlreadyExists,
                    observation: existing,
                });
            } else {
                // Different contents: reject with explicit conflict error
                return Err(StorageError::DuplicateIdConflict {
                    id: obs.id.to_string(),
                });
            }
        }

        // Insert observation row
        let statement_payload_str = serde_json::to_string(&obs.statement.payload)?;
        let created_at_rfc3339 = obs.created_at.to_rfc3339();

        tx.execute(
            r#"
            INSERT INTO observations (id, author, subject, statement_kind, statement_payload, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                obs.id.as_str(),
                obs.author.as_str(),
                &obs.subject,
                &obs.statement.kind,
                &statement_payload_str,
                &created_at_rfc3339,
            ],
        )?;

        // Insert evidence references
        for ev in &obs.evidence {
            let kind_str = match &ev.kind {
                EvidenceKind::GitCommit => "git_commit",
                EvidenceKind::FileDigest => "file_digest",
                EvidenceKind::Uri => "uri",
                EvidenceKind::ExecutionTrace => "execution_trace",
                EvidenceKind::ExternalReport => "external_report",
                EvidenceKind::Other(k) => k.as_str(),
            };
            tx.execute(
                r#"
                INSERT INTO observation_evidence (observation_id, kind, locator, description)
                VALUES (?1, ?2, ?3, ?4)
                "#,
                params![obs.id.as_str(), kind_str, &ev.locator, &ev.description],
            )?;
        }

        tx.commit()?;

        Ok(PublishResult {
            status: PublishStatus::Created,
            observation: obs,
        })
    }

    /// Retrieves observations using bounded pagination and optional filters in deterministic order.
    pub fn read_observations(
        &self,
        filter: &ObservationFilter,
    ) -> Result<ObservationPage, StorageError> {
        let conn = self.conn.lock().unwrap();

        let limit = filter.bounded_limit();
        let offset = filter.offset;

        let mut where_clauses = Vec::new();
        let mut count_params: Vec<rusqlite::types::Value> = Vec::new();

        if let Some(ref author) = filter.author {
            where_clauses.push(format!("author = ?{}", count_params.len() + 1));
            count_params.push(author.as_str().to_string().into());
        }
        if let Some(ref subject) = filter.subject {
            where_clauses.push(format!("subject = ?{}", count_params.len() + 1));
            count_params.push(subject.clone().into());
        }
        if let Some(ref since) = filter.since {
            where_clauses.push(format!("created_at >= ?{}", count_params.len() + 1));
            count_params.push(since.to_rfc3339().into());
        }

        let where_sql = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        // Query total count
        let count_sql = format!("SELECT COUNT(*) FROM observations {where_sql}");
        let total_count: usize = conn.query_row(
            &count_sql,
            rusqlite::params_from_iter(count_params.iter()),
            |r| r.get::<_, i64>(0).map(|c| c as usize),
        )?;

        // Query paginated observations with deterministic ordering
        let query_sql = format!(
            "SELECT id, author, subject, statement_kind, statement_payload, created_at
             FROM observations
             {where_sql}
             ORDER BY created_at ASC, id ASC
             LIMIT ?{} OFFSET ?{}",
            count_params.len() + 1,
            count_params.len() + 2
        );

        let mut query_params = count_params;
        query_params.push((limit as i64).into());
        query_params.push((offset as i64).into());

        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt.query_map(rusqlite::params_from_iter(query_params.iter()), |row| {
            let id: String = row.get(0)?;
            let author: String = row.get(1)?;
            let subject: String = row.get(2)?;
            let statement_kind: String = row.get(3)?;
            let statement_payload: String = row.get(4)?;
            let created_at: String = row.get(5)?;
            Ok((
                id,
                author,
                subject,
                statement_kind,
                statement_payload,
                created_at,
            ))
        })?;

        let mut obs_records = Vec::new();
        for r in rows {
            obs_records.push(r?);
        }

        // Fetch evidence for each observation
        let mut items = Vec::new();
        for (id_str, author_str, subject, stmt_kind, stmt_payload_str, created_at_str) in
            obs_records
        {
            let id = ObservationId::new(id_str)?;
            let author = ParticipantId::new(author_str)?;
            let payload: serde_json::Value = serde_json::from_str(&stmt_payload_str)?;
            let statement = ObservationStatement::new(stmt_kind, payload)?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|e| StorageError::Migration(e.to_string()))?
                .with_timezone(&Utc);

            let evidence = load_evidence_for_observation(&conn, id.as_str())?;

            items.push(Observation::new(
                id, author, subject, statement, evidence, created_at,
            )?);
        }

        let has_more = offset + items.len() < total_count;

        Ok(ObservationPage {
            items,
            total_count,
            limit,
            offset,
            has_more,
        })
    }

    fn seed_built_in_capabilities(&self) -> Result<(), StorageError> {
        let provider = ParticipantId::new("system:huerarchy")?;
        let epoch = Utc::now();

        let capabilities = [
            (
                "cap:health",
                "health",
                "Returns substrate service readiness, protocol and schema versions, and storage metrics.",
                "0.1.0",
                serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            ),
            (
                "cap:capabilities.list",
                "list_capabilities",
                "Discovers all registered computational capabilities and their documented contracts.",
                "0.1.0",
                serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            ),
            (
                "cap:observations.publish",
                "publish_observation",
                "Records a typed, durable observation with author, subject, statement payload, and provenance evidence.",
                "0.1.0",
                serde_json::json!({
                    "type": "object",
                    "required": ["author", "subject", "statement"],
                    "properties": {
                        "author": { "type": "string" },
                        "subject": { "type": "string" },
                        "statement": {
                            "type": "object",
                            "required": ["kind", "payload"],
                            "properties": {
                                "kind": { "type": "string" },
                                "payload": {}
                            }
                        },
                        "evidence": { "type": "array" },
                        "id": { "type": "string" }
                    }
                }),
            ),
            (
                "cap:observations.read",
                "read_observations",
                "Retrieves durable observations using bounded pagination and optional author, subject, or timestamp filters.",
                "0.1.0",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "author": { "type": "string" },
                        "subject": { "type": "string" },
                        "since": { "type": "string" },
                        "limit": { "type": "integer", "minimum": 1, "maximum": 100 },
                        "offset": { "type": "integer", "minimum": 0 }
                    }
                }),
            ),
        ];

        for (id, name, desc, ver, schema) in capabilities {
            let cap = Capability::new(
                CapabilityId::new(id)?,
                name,
                desc,
                ver,
                schema,
                provider.clone(),
                epoch,
            )?;
            self.register_capability(&cap)?;
        }

        Ok(())
    }
}

fn configure_connection(conn: &mut Connection, is_memory: bool) -> Result<(), StorageError> {
    if !is_memory {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
    }
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.busy_timeout(std::time::Duration::from_millis(5000))?;
    Ok(())
}

fn load_evidence_for_observation(
    conn: &Connection,
    obs_id: &str,
) -> Result<Vec<EvidenceRef>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT kind, locator, description
         FROM observation_evidence
         WHERE observation_id = ?1
         ORDER BY id ASC",
    )?;

    let rows = stmt.query_map([obs_id], |row| {
        let kind_str: String = row.get(0)?;
        let locator: String = row.get(1)?;
        let description: Option<String> = row.get(2)?;
        Ok((kind_str, locator, description))
    })?;

    let mut evidence = Vec::new();
    for r in rows {
        let (kind_str, locator, description) = r?;
        let kind = match kind_str.as_str() {
            "git_commit" => EvidenceKind::GitCommit,
            "file_digest" => EvidenceKind::FileDigest,
            "uri" => EvidenceKind::Uri,
            "execution_trace" => EvidenceKind::ExecutionTrace,
            "external_report" => EvidenceKind::ExternalReport,
            other => EvidenceKind::Other(other.to_string()),
        };
        evidence.push(EvidenceRef::new(kind, locator, description)?);
    }

    Ok(evidence)
}

fn load_observation_by_id(
    conn: &Connection,
    obs_id: &str,
) -> Result<Option<Observation>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT id, author, subject, statement_kind, statement_payload, created_at
         FROM observations
         WHERE id = ?1",
    )?;

    let row = stmt
        .query_row([obs_id], |row| {
            let id: String = row.get(0)?;
            let author: String = row.get(1)?;
            let subject: String = row.get(2)?;
            let statement_kind: String = row.get(3)?;
            let statement_payload: String = row.get(4)?;
            let created_at: String = row.get(5)?;
            Ok((
                id,
                author,
                subject,
                statement_kind,
                statement_payload,
                created_at,
            ))
        })
        .optional()?;

    match row {
        Some((id_str, author_str, subject, stmt_kind, stmt_payload_str, created_at_str)) => {
            let id = ObservationId::new(id_str)?;
            let author = ParticipantId::new(author_str)?;
            let payload: serde_json::Value = serde_json::from_str(&stmt_payload_str)?;
            let statement = ObservationStatement::new(stmt_kind, payload)?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|e| StorageError::Migration(e.to_string()))?
                .with_timezone(&Utc);

            let evidence = load_evidence_for_observation(conn, id.as_str())?;

            Ok(Some(Observation::new(
                id, author, subject, statement, evidence, created_at,
            )?))
        }
        None => Ok(None),
    }
}
