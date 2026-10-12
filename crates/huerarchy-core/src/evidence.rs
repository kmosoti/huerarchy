use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::DomainError;

const MAX_LOCATOR_LEN: usize = 1024;
const MAX_DESC_LEN: usize = 512;

/// Classification of evidence supporting an observation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    GitCommit,
    FileDigest,
    Uri,
    ExecutionTrace,
    ExternalReport,
    Other(String),
}

/// A reference to external or cryptographic evidence grounding an observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceRef {
    pub kind: EvidenceKind,
    pub locator: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl EvidenceRef {
    pub fn new(
        kind: EvidenceKind,
        locator: impl Into<String>,
        description: Option<String>,
    ) -> Result<Self, DomainError> {
        let locator = locator.into();
        if locator.trim().is_empty() {
            return Err(DomainError::EmptyField("evidence.locator"));
        }
        if locator.len() > MAX_LOCATOR_LEN {
            return Err(DomainError::FieldTooLong {
                field: "evidence.locator",
                max: MAX_LOCATOR_LEN,
                actual: locator.len(),
            });
        }

        if let Some(ref desc) = description {
            if desc.len() > MAX_DESC_LEN {
                return Err(DomainError::FieldTooLong {
                    field: "evidence.description",
                    max: MAX_DESC_LEN,
                    actual: desc.len(),
                });
            }
        }

        Ok(Self {
            kind,
            locator,
            description,
        })
    }
}
