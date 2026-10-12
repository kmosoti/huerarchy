use huerarchy_core::{
    Capability, EvidenceKind, EvidenceRef, Observation, ObservationFilter, ParticipantId,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::storage::{PublishStatus, StorageHealth};

/// Parameters for the publish_observation tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PublishObservationParams {
    /// Identifier of the observing participant (e.g. `agent:claude-code`, `human:kmosoti`).
    pub author: String,

    /// Target entity or subject being observed (e.g. `repo:huerarchy`, `node:fedora-blade`).
    pub subject: String,

    /// Typed statement payload.
    pub statement: StatementInput,

    /// Supporting provenance references.
    #[serde(default)]
    pub evidence: Vec<EvidenceInput>,

    /// Optional stable identifier. If omitted, a UUIDv4 will be generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StatementInput {
    /// Statement type discriminator.
    pub kind: String,
    /// Arbitrary structured JSON data payload.
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceInput {
    /// Evidence category: `git_commit`, `file_digest`, `uri`, `execution_trace`, `external_report`, or custom.
    pub kind: String,
    /// Locator string (e.g. commit sha, URI, content hash).
    pub locator: String,
    /// Optional summary or human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl EvidenceInput {
    pub fn into_evidence_ref(self) -> Result<EvidenceRef, huerarchy_core::DomainError> {
        let kind = match self.kind.as_str() {
            "git_commit" => EvidenceKind::GitCommit,
            "file_digest" => EvidenceKind::FileDigest,
            "uri" => EvidenceKind::Uri,
            "execution_trace" => EvidenceKind::ExecutionTrace,
            "external_report" => EvidenceKind::ExternalReport,
            other => EvidenceKind::Other(other.to_string()),
        };
        EvidenceRef::new(kind, self.locator, self.description)
    }
}

/// Parameters for the read_observations tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ReadObservationsParams {
    /// Optional filter by author participant ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,

    /// Optional filter by subject.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,

    /// Optional RFC 3339 UTC timestamp filter (returns observations >= since).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<String>,

    /// Maximum number of records to return (1 to 100, default 50).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,

    /// Number of records to skip for pagination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
}

impl ReadObservationsParams {
    pub fn into_filter(self) -> Result<ObservationFilter, huerarchy_core::DomainError> {
        let author = match self.author {
            Some(a) => Some(ParticipantId::new(a)?),
            None => None,
        };

        let since = match self.since {
            Some(s) => {
                let dt = chrono::DateTime::parse_from_rfc3339(&s)
                    .map_err(|e| huerarchy_core::DomainError::InvalidTimestamp(e.to_string()))?
                    .with_timezone(&chrono::Utc);
                Some(dt)
            }
            None => None,
        };

        Ok(ObservationFilter {
            author,
            subject: self.subject,
            since,
            limit: self.limit.unwrap_or(huerarchy_core::DEFAULT_PAGE_LIMIT),
            offset: self.offset.unwrap_or(0),
        })
    }
}

/// Structured response for the health tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HealthResponse {
    pub status: String,
    pub protocol_version: String,
    pub schema_version: i32,
    pub storage: StorageHealth,
}

/// Structured response for the list_capabilities tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListCapabilitiesResponse {
    pub capabilities: Vec<Capability>,
}

/// Structured response for the publish_observation tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PublishObservationResponse {
    pub status: PublishStatus,
    pub observation: Observation,
}

/// Structured response for the read_observations tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReadObservationsResponse {
    pub items: Vec<Observation>,
    pub total_count: usize,
    pub limit: usize,
    pub offset: usize,
    pub has_more: bool,
}

/// Structured error payload.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ErrorPayload {
    pub error: String,
    pub details: Option<String>,
}
