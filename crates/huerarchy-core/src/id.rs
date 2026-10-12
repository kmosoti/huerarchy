use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::DomainError;

const MAX_ID_LEN: usize = 128;

fn validate_identifier(field: &'static str, s: &str) -> Result<(), DomainError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(DomainError::EmptyField(field));
    }
    if s.len() != trimmed.len() {
        return Err(DomainError::InvalidFormat {
            field,
            reason: "identifier must not contain leading or trailing whitespace".to_string(),
        });
    }
    if s.len() > MAX_ID_LEN {
        return Err(DomainError::FieldTooLong {
            field,
            max: MAX_ID_LEN,
            actual: s.len(),
        });
    }
    // Permitted characters: ASCII alphanumeric, '-', '_', ':', '.', '/'
    for c in s.chars() {
        if !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != ':' && c != '.' && c != '/' {
            return Err(DomainError::InvalidFormat {
                field,
                reason: format!(
                    "character '{c}' is not permitted; use ASCII alphanumeric or [- _ : . /]"
                ),
            });
        }
    }
    Ok(())
}

/// Strongly typed, validated participant identifier.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct ParticipantId(String);

impl ParticipantId {
    pub fn new(id: impl Into<String>) -> Result<Self, DomainError> {
        let s = id.into();
        validate_identifier("participant_id", &s)?;
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ParticipantId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ParticipantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ParticipantId {
    type Err = DomainError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Strongly typed, validated capability identifier.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct CapabilityId(String);

impl CapabilityId {
    pub fn new(id: impl Into<String>) -> Result<Self, DomainError> {
        let s = id.into();
        validate_identifier("capability_id", &s)?;
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for CapabilityId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CapabilityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for CapabilityId {
    type Err = DomainError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Strongly typed, validated observation identifier.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct ObservationId(String);

impl ObservationId {
    pub fn new(id: impl Into<String>) -> Result<Self, DomainError> {
        let s = id.into();
        validate_identifier("observation_id", &s)?;
        Ok(Self(s))
    }

    /// Generates a random UUIDv4 observation identifier.
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ObservationId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ObservationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for ObservationId {
    type Err = DomainError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}
