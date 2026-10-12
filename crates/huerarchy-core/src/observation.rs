use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::DomainError;
use crate::evidence::EvidenceRef;
use crate::id::{ObservationId, ParticipantId};

const MAX_SUBJECT_LEN: usize = 256;
const MAX_STATEMENT_KIND_LEN: usize = 64;
const MAX_STATEMENT_PAYLOAD_BYTES: usize = 65536; // 64 KiB
const MAX_EVIDENCE_COUNT: usize = 16;
pub const DEFAULT_PAGE_LIMIT: usize = 50;
pub const MAX_PAGE_LIMIT: usize = 100;

/// The typed statement content of an observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ObservationStatement {
    /// Statement type discriminator (e.g. `system_health`, `test_result`, `metric`).
    pub kind: String,
    /// Arbitrary structured JSON data payload.
    pub payload: serde_json::Value,
}

impl ObservationStatement {
    pub fn new(kind: impl Into<String>, payload: serde_json::Value) -> Result<Self, DomainError> {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(DomainError::EmptyField("statement.kind"));
        }
        if kind.len() > MAX_STATEMENT_KIND_LEN {
            return Err(DomainError::FieldTooLong {
                field: "statement.kind",
                max: MAX_STATEMENT_KIND_LEN,
                actual: kind.len(),
            });
        }

        // Validate payload size when serialized
        let serialized =
            serde_json::to_string(&payload).map_err(|e| DomainError::InvalidJson(e.to_string()))?;
        if serialized.len() > MAX_STATEMENT_PAYLOAD_BYTES {
            return Err(DomainError::PayloadTooLarge {
                max_bytes: MAX_STATEMENT_PAYLOAD_BYTES,
                actual_bytes: serialized.len(),
            });
        }

        Ok(Self { kind, payload })
    }
}

/// A durable, typed observation recorded by a participant in the substrate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Observation {
    /// Unique identifier for this observation.
    pub id: ObservationId,
    /// Author / observer participant.
    pub author: ParticipantId,
    /// Target subject of the observation (e.g. `repo:huerarchy`, `agent:claude`).
    pub subject: String,
    /// Typed statement payload.
    pub statement: ObservationStatement,
    /// Supporting evidence references for provenance.
    pub evidence: Vec<EvidenceRef>,
    /// UTC timestamp when observation occurred.
    pub created_at: DateTime<Utc>,
}

impl Observation {
    pub fn new(
        id: ObservationId,
        author: ParticipantId,
        subject: impl Into<String>,
        statement: ObservationStatement,
        evidence: Vec<EvidenceRef>,
        created_at: DateTime<Utc>,
    ) -> Result<Self, DomainError> {
        let subject = subject.into();
        if subject.trim().is_empty() {
            return Err(DomainError::EmptyField("observation.subject"));
        }
        if subject.len() > MAX_SUBJECT_LEN {
            return Err(DomainError::FieldTooLong {
                field: "observation.subject",
                max: MAX_SUBJECT_LEN,
                actual: subject.len(),
            });
        }

        if evidence.len() > MAX_EVIDENCE_COUNT {
            return Err(DomainError::TooManyItems {
                field: "observation.evidence",
                max: MAX_EVIDENCE_COUNT,
                actual: evidence.len(),
            });
        }

        Ok(Self {
            id,
            author,
            subject,
            statement,
            evidence,
            created_at,
        })
    }
}

/// Query filter parameters for reading observations with bounded pagination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ObservationFilter {
    /// Filter by author participant ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<ParticipantId>,

    /// Filter by subject.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,

    /// Filter for observations recorded at or after this timestamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<DateTime<Utc>>,

    /// Maximum number of records to return (clamped between 1 and 100).
    #[serde(default = "default_page_limit")]
    pub limit: usize,

    /// Offset for pagination.
    #[serde(default)]
    pub offset: usize,
}

fn default_page_limit() -> usize {
    DEFAULT_PAGE_LIMIT
}

impl Default for ObservationFilter {
    fn default() -> Self {
        Self {
            author: None,
            subject: None,
            since: None,
            limit: DEFAULT_PAGE_LIMIT,
            offset: 0,
        }
    }
}

impl ObservationFilter {
    /// Normalized page limit bounded within safe limits.
    pub fn bounded_limit(&self) -> usize {
        if self.limit == 0 {
            DEFAULT_PAGE_LIMIT
        } else {
            self.limit.min(MAX_PAGE_LIMIT)
        }
    }
}

/// Bounded pagination result for observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ObservationPage {
    pub items: Vec<Observation>,
    pub total_count: usize,
    pub limit: usize,
    pub offset: usize,
    pub has_more: bool,
}
