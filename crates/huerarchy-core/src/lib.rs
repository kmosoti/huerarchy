pub mod capability;
pub mod error;
pub mod evidence;
pub mod id;
pub mod observation;
pub mod participant;

pub use capability::Capability;
pub use error::DomainError;
pub use evidence::{EvidenceKind, EvidenceRef};
pub use id::{CapabilityId, ObservationId, ParticipantId};
pub use observation::{
    Observation, ObservationFilter, ObservationPage, ObservationStatement, DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
};
pub use participant::{Participant, ParticipantKind};

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn test_participant_id_validation() {
        assert!(ParticipantId::new("agent:claude-code").is_ok());
        assert!(ParticipantId::new("agent:codex").is_ok());
        assert!(ParticipantId::new("human:kmosoti").is_ok());
        assert!(ParticipantId::new("system.local_node_1").is_ok());

        // Invalid cases
        assert!(ParticipantId::new("").is_err());
        assert!(ParticipantId::new("   ").is_err());
        assert!(ParticipantId::new("agent with spaces").is_err());
        assert!(ParticipantId::new("agent#bad_char").is_err());
        assert!(ParticipantId::new("a".repeat(129)).is_err());
    }

    #[test]
    fn test_capability_validation() {
        let cap_id = CapabilityId::new("cap:health").unwrap();
        let provider = ParticipantId::new("system:node").unwrap();
        let schema = json!({
            "type": "object",
            "properties": {
                "status": { "type": "string" }
            }
        });

        let cap = Capability::new(
            cap_id,
            "Health Check",
            "Returns system readiness status",
            "0.1.0",
            schema,
            provider,
            Utc::now(),
        );
        assert!(cap.is_ok());

        // Reject empty fields
        assert!(Capability::new(
            CapabilityId::new("cap:invalid").unwrap(),
            "",
            "desc",
            "1.0",
            json!({}),
            ParticipantId::new("sys").unwrap(),
            Utc::now()
        )
        .is_err());

        // Reject non-object/non-boolean schema
        assert!(Capability::new(
            CapabilityId::new("cap:invalid").unwrap(),
            "Name",
            "desc",
            "1.0",
            json!("not a schema"),
            ParticipantId::new("sys").unwrap(),
            Utc::now()
        )
        .is_err());
    }

    #[test]
    fn test_evidence_validation() {
        assert!(EvidenceRef::new(
            EvidenceKind::GitCommit,
            "b940084d72026092800000000000000000000000",
            Some("Initial commit sha".to_string())
        )
        .is_ok());

        assert!(EvidenceRef::new(EvidenceKind::GitCommit, "", None).is_err());

        assert!(EvidenceRef::new(
            EvidenceKind::Uri,
            "file:///var/log/test.log",
            Some("x".repeat(600)) // exceeds 512
        )
        .is_err());
    }

    #[test]
    fn test_observation_validation_and_bounds() {
        let id = ObservationId::generate();
        let author = ParticipantId::new("agent:antigravity").unwrap();
        let statement = ObservationStatement::new(
            "test_result",
            json!({ "suite": "core_invariants", "passed": 12, "failed": 0 }),
        )
        .unwrap();

        let obs = Observation::new(
            id,
            author.clone(),
            "repo:huerarchy",
            statement,
            vec![],
            Utc::now(),
        );
        assert!(obs.is_ok());

        // Reject empty subject
        let stmt = ObservationStatement::new("kind", json!({})).unwrap();
        assert!(Observation::new(
            ObservationId::generate(),
            author.clone(),
            "  ",
            stmt.clone(),
            vec![],
            Utc::now()
        )
        .is_err());

        // Reject excessive evidence references (> 16)
        let too_much_evidence: Vec<EvidenceRef> = (0..17)
            .map(|i| {
                EvidenceRef::new(EvidenceKind::ExecutionTrace, format!("trace-{i}"), None).unwrap()
            })
            .collect();
        assert!(Observation::new(
            ObservationId::generate(),
            author,
            "subject",
            stmt,
            too_much_evidence,
            Utc::now()
        )
        .is_err());
    }

    #[test]
    fn test_observation_statement_payload_bounds() {
        // Large payload exceeding 64 KiB should be rejected
        let large_string = "a".repeat(70000);
        let payload = json!({ "blob": large_string });
        let result = ObservationStatement::new("large_blob", payload);
        assert!(matches!(result, Err(DomainError::PayloadTooLarge { .. })));
    }

    #[test]
    fn test_observation_filter_limit_bounding() {
        let mut filter = ObservationFilter::default();
        assert_eq!(filter.bounded_limit(), DEFAULT_PAGE_LIMIT);

        filter.limit = 200;
        assert_eq!(filter.bounded_limit(), MAX_PAGE_LIMIT);

        filter.limit = 10;
        assert_eq!(filter.bounded_limit(), 10);

        filter.limit = 0;
        assert_eq!(filter.bounded_limit(), DEFAULT_PAGE_LIMIT);
    }
}
