use huerarchy_core::DomainError;
use thiserror::Error;

/// Storage and persistence errors.
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("domain error: {0}")]
    Domain(#[from] DomainError),

    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("filesystem I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("database migration error: {0}")]
    Migration(String),

    #[error("duplicate observation conflict: an observation with id '{id}' already exists with different contents")]
    DuplicateIdConflict { id: String },

    #[error("path error: {0}")]
    PathError(String),
}
