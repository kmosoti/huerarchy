use chrono::Utc;
use rusqlite::{Connection, Transaction};

use crate::storage::error::StorageError;

pub const CURRENT_SCHEMA_VERSION: i32 = 1;

/// Runs all pending database migrations in sequence inside transactions.
pub fn run_migrations(conn: &mut Connection) -> Result<i32, StorageError> {
    let current_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;

    if current_version > CURRENT_SCHEMA_VERSION {
        return Err(StorageError::Migration(format!(
            "Database schema version {current_version} is newer than supported version {CURRENT_SCHEMA_VERSION}"
        )));
    }

    if current_version < 1 {
        let tx = conn.transaction()?;
        migrate_v1(&tx)?;
        tx.pragma_update(None, "user_version", 1)?;
        tx.commit()?;
    }

    let final_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    Ok(final_version)
}

fn migrate_v1(tx: &Transaction) -> Result<(), StorageError> {
    tx.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS _huerarchy_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS capabilities (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            version TEXT NOT NULL,
            contract_schema TEXT NOT NULL,
            provider TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY NOT NULL,
            author TEXT NOT NULL,
            subject TEXT NOT NULL,
            statement_kind TEXT NOT NULL,
            statement_payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS observation_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            observation_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            locator TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (observation_id) REFERENCES observations(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_observations_created_at_id
            ON observations(created_at ASC, id ASC);

        CREATE INDEX IF NOT EXISTS idx_observations_author
            ON observations(author);

        CREATE INDEX IF NOT EXISTS idx_observations_subject
            ON observations(subject);

        CREATE INDEX IF NOT EXISTS idx_observation_evidence_obs_id
            ON observation_evidence(observation_id);
        "#,
    )?;

    let now = Utc::now().to_rfc3339();
    tx.execute(
        "INSERT INTO _huerarchy_migrations (version, applied_at) VALUES (?1, ?2)",
        (1, &now),
    )?;

    Ok(())
}
