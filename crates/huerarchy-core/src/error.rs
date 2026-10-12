use thiserror::Error;

/// Errors arising from domain invariant violations.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum DomainError {
    #[error("field '{0}' cannot be empty")]
    EmptyField(&'static str),

    #[error("field '{field}' exceeds maximum length of {max} characters (got {actual})")]
    FieldTooLong {
        field: &'static str,
        max: usize,
        actual: usize,
    },

    #[error("field '{field}' contains invalid characters or format: {reason}")]
    InvalidFormat { field: &'static str, reason: String },

    #[error("payload exceeds maximum size of {max_bytes} bytes (got {actual_bytes} bytes)")]
    PayloadTooLarge {
        max_bytes: usize,
        actual_bytes: usize,
    },

    #[error("list '{field}' exceeds maximum item count of {max} (got {actual})")]
    TooManyItems {
        field: &'static str,
        max: usize,
        actual: usize,
    },

    #[error("invalid timestamp '{0}': must be valid RFC 3339 UTC timestamp")]
    InvalidTimestamp(String),

    #[error("invalid JSON value: {0}")]
    InvalidJson(String),
}
