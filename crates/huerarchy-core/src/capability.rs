use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::DomainError;
use crate::id::{CapabilityId, ParticipantId};

const MAX_NAME_LEN: usize = 128;
const MAX_DESC_LEN: usize = 1024;
const MAX_VERSION_LEN: usize = 64;

/// A documented, reusable computational capability contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Capability {
    pub id: CapabilityId,
    pub name: String,
    pub description: String,
    pub version: String,
    pub contract_schema: serde_json::Value,
    pub provider: ParticipantId,
    pub created_at: DateTime<Utc>,
}

impl Capability {
    pub fn new(
        id: CapabilityId,
        name: impl Into<String>,
        description: impl Into<String>,
        version: impl Into<String>,
        contract_schema: serde_json::Value,
        provider: ParticipantId,
        created_at: DateTime<Utc>,
    ) -> Result<Self, DomainError> {
        let name = name.into();
        let description = description.into();
        let version = version.into();

        if name.trim().is_empty() {
            return Err(DomainError::EmptyField("capability.name"));
        }
        if name.len() > MAX_NAME_LEN {
            return Err(DomainError::FieldTooLong {
                field: "capability.name",
                max: MAX_NAME_LEN,
                actual: name.len(),
            });
        }

        if description.trim().is_empty() {
            return Err(DomainError::EmptyField("capability.description"));
        }
        if description.len() > MAX_DESC_LEN {
            return Err(DomainError::FieldTooLong {
                field: "capability.description",
                max: MAX_DESC_LEN,
                actual: description.len(),
            });
        }

        if version.trim().is_empty() {
            return Err(DomainError::EmptyField("capability.version"));
        }
        if version.len() > MAX_VERSION_LEN {
            return Err(DomainError::FieldTooLong {
                field: "capability.version",
                max: MAX_VERSION_LEN,
                actual: version.len(),
            });
        }

        if !contract_schema.is_object() && !contract_schema.is_boolean() {
            return Err(DomainError::InvalidJson(
                "contract_schema must be a valid JSON Schema object or boolean".to_string(),
            ));
        }

        Ok(Self {
            id,
            name,
            description,
            version,
            contract_schema,
            provider,
            created_at,
        })
    }
}
