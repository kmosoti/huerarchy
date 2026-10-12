use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::DomainError;
use crate::id::ParticipantId;

const MAX_NAME_LEN: usize = 128;

/// Category of participant in the Huerarchy substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ParticipantKind {
    Human,
    Agent,
    System,
}

/// A registered participant capable of observing or providing capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Participant {
    pub id: ParticipantId,
    pub kind: ParticipantKind,
    pub name: String,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Participant {
    pub fn new(
        id: ParticipantId,
        kind: ParticipantKind,
        name: impl Into<String>,
        created_at: DateTime<Utc>,
        metadata: Option<serde_json::Value>,
    ) -> Result<Self, DomainError> {
        let name = name.into();
        let trimmed_name = name.trim();
        if trimmed_name.is_empty() {
            return Err(DomainError::EmptyField("participant.name"));
        }
        if name.len() > MAX_NAME_LEN {
            return Err(DomainError::FieldTooLong {
                field: "participant.name",
                max: MAX_NAME_LEN,
                actual: name.len(),
            });
        }
        Ok(Self {
            id,
            kind,
            name,
            created_at,
            metadata,
        })
    }
}
