use std::thread;

use chrono::Utc;
use huerarchy_core::{
    EvidenceKind, EvidenceRef, Observation, ObservationFilter, ObservationId, ObservationStatement,
    ParticipantId,
};
use tempfile::NamedTempFile;

use crate::storage::{PublishStatus, Storage, StorageError};

#[test]
fn test_storage_open_and_health() {
    let storage = Storage::open_in_memory().expect("in-memory db opens");
    let health = storage.health().expect("health check succeeds");
    assert_eq!(health.schema_version, 1);
    assert_eq!(health.observation_count, 0);
    assert!(health.capability_count >= 4);
}

#[test]
fn test_publish_and_read_observations() {
    let storage = Storage::open_in_memory().unwrap();
    let author = ParticipantId::new("agent:claude-code").unwrap();
    let id = ObservationId::generate();
    let stmt = ObservationStatement::new(
        "code_analysis",
        serde_json::json!({ "files_analyzed": 14, "issues": 0 }),
    )
    .unwrap();
    let evidence = vec![EvidenceRef::new(
        EvidenceKind::GitCommit,
        "abcdef1234567890abcdef1234567890abcdef12",
        Some("commit test".to_string()),
    )
    .unwrap()];

    let obs = Observation::new(
        id.clone(),
        author.clone(),
        "repo:huerarchy",
        stmt,
        evidence,
        Utc::now(),
    )
    .unwrap();

    let publish_res = storage.publish_observation(obs.clone()).unwrap();
    assert_eq!(publish_res.status, PublishStatus::Created);
    assert_eq!(publish_res.observation.id, id);

    // Read back
    let page = storage
        .read_observations(&ObservationFilter::default())
        .unwrap();
    assert_eq!(page.total_count, 1);
    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0], obs);
}

#[test]
fn test_observation_idempotency_policy() {
    let storage = Storage::open_in_memory().unwrap();
    let author = ParticipantId::new("agent:codex").unwrap();
    let id = ObservationId::generate();
    let stmt1 = ObservationStatement::new("metric", serde_json::json!({ "val": 100 })).unwrap();
    let obs1 = Observation::new(
        id.clone(),
        author.clone(),
        "perf:eval",
        stmt1,
        vec![],
        Utc::now(),
    )
    .unwrap();

    // First publish: Created
    let res1 = storage.publish_observation(obs1.clone()).unwrap();
    assert_eq!(res1.status, PublishStatus::Created);

    // Second publish with exact same data: AlreadyExists (idempotent replay)
    let res2 = storage.publish_observation(obs1.clone()).unwrap();
    assert_eq!(res2.status, PublishStatus::AlreadyExists);
    assert_eq!(res2.observation, obs1);

    // Third publish with same ID but conflicting data: Rejected with DuplicateIdConflict
    let stmt2 = ObservationStatement::new("metric", serde_json::json!({ "val": 200 })).unwrap();
    let conflicting_obs =
        Observation::new(id.clone(), author, "perf:eval", stmt2, vec![], Utc::now()).unwrap();

    let err = storage.publish_observation(conflicting_obs).unwrap_err();
    assert!(matches!(err, StorageError::DuplicateIdConflict { .. }));
}

#[test]
fn test_persistence_across_restarts() {
    let temp_file = NamedTempFile::new().unwrap();
    let db_path = temp_file.path().to_path_buf();

    let obs_id = ObservationId::generate();
    let author = ParticipantId::new("human:kmosoti").unwrap();
    let stmt = ObservationStatement::new(
        "bootstrap_milestone",
        serde_json::json!({ "status": "foundation_v0.1" }),
    )
    .unwrap();
    let obs = Observation::new(
        obs_id.clone(),
        author,
        "huerarchy:node",
        stmt,
        vec![],
        Utc::now(),
    )
    .unwrap();

    // First session: open, write, drop
    {
        let storage = Storage::open(&db_path).unwrap();
        let res = storage.publish_observation(obs.clone()).unwrap();
        assert_eq!(res.status, PublishStatus::Created);
        let health = storage.health().unwrap();
        assert_eq!(health.observation_count, 1);
        assert_eq!(health.journal_mode.to_uppercase(), "WAL");
    } // storage dropped here

    // Second session: re-open from same file, verify data intact
    {
        let reopened = Storage::open(&db_path).unwrap();
        let health = reopened.health().unwrap();
        assert_eq!(health.observation_count, 1);

        let page = reopened
            .read_observations(&ObservationFilter::default())
            .unwrap();
        assert_eq!(page.total_count, 1);
        assert_eq!(page.items[0], obs);
    }
}

#[test]
fn test_concurrent_writes_preserve_consistency() {
    let temp_file = NamedTempFile::new().unwrap();
    let db_path = temp_file.path().to_path_buf();

    // Initialize schema
    {
        let _ = Storage::open(&db_path).unwrap();
    }

    let num_threads = 4;
    let writes_per_thread = 25;
    let mut handles = Vec::new();

    for t in 0..num_threads {
        let path = db_path.clone();
        let handle = thread::spawn(move || {
            let storage = Storage::open(&path).expect("open inside thread");
            let author = ParticipantId::new(format!("agent:worker-{t}")).unwrap();
            for i in 0..writes_per_thread {
                let obs = Observation::new(
                    ObservationId::generate(),
                    author.clone(),
                    format!("subject:task-{t}-{i}"),
                    ObservationStatement::new("ping", serde_json::json!({ "step": i })).unwrap(),
                    vec![],
                    Utc::now(),
                )
                .unwrap();
                storage.publish_observation(obs).expect("write succeeds");
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify all writes survived
    let verify_storage = Storage::open(&db_path).unwrap();
    let health = verify_storage.health().unwrap();
    assert_eq!(health.observation_count, num_threads * writes_per_thread);

    let page_partial = verify_storage
        .read_observations(&ObservationFilter {
            limit: 50,
            offset: 0,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(page_partial.items.len(), 50);
    assert_eq!(page_partial.total_count, num_threads * writes_per_thread);
    assert!(page_partial.has_more);

    let page_all = verify_storage
        .read_observations(&ObservationFilter {
            limit: 100,
            offset: 0,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(page_all.items.len(), 100);
    assert!(!page_all.has_more);
}

#[test]
fn test_observation_pagination_and_deterministic_order() {
    let storage = Storage::open_in_memory().unwrap();
    let author = ParticipantId::new("agent:tester").unwrap();

    let base_time = Utc::now();
    for i in 0..10 {
        let obs = Observation::new(
            ObservationId::new(format!("obs-{:02}", i)).unwrap(),
            author.clone(),
            "repo:order_test",
            ObservationStatement::new("seq", serde_json::json!({ "idx": i })).unwrap(),
            vec![],
            base_time + chrono::Duration::seconds(i as i64),
        )
        .unwrap();
        storage.publish_observation(obs).unwrap();
    }

    // Page 1: limit 4, offset 0
    let p1 = storage
        .read_observations(&ObservationFilter {
            limit: 4,
            offset: 0,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(p1.items.len(), 4);
    assert_eq!(p1.items[0].id.as_str(), "obs-00");
    assert_eq!(p1.items[3].id.as_str(), "obs-03");
    assert!(p1.has_more);

    // Page 2: limit 4, offset 4
    let p2 = storage
        .read_observations(&ObservationFilter {
            limit: 4,
            offset: 4,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(p2.items.len(), 4);
    assert_eq!(p2.items[0].id.as_str(), "obs-04");
    assert_eq!(p2.items[3].id.as_str(), "obs-07");
    assert!(p2.has_more);

    // Page 3: limit 4, offset 8
    let p3 = storage
        .read_observations(&ObservationFilter {
            limit: 4,
            offset: 8,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(p3.items.len(), 2);
    assert_eq!(p3.items[0].id.as_str(), "obs-08");
    assert_eq!(p3.items[1].id.as_str(), "obs-09");
    assert!(!p3.has_more);
}

#[test]
fn test_database_migrations_preserve_data() {
    let temp_file = NamedTempFile::new().unwrap();
    let db_path = temp_file.path().to_path_buf();

    let author = ParticipantId::new("agent:antigravity").unwrap();
    let obs_id = ObservationId::generate();
    let obs = Observation::new(
        obs_id.clone(),
        author,
        "migration_check",
        ObservationStatement::new("invariant", serde_json::json!({ "pre_migrated": true }))
            .unwrap(),
        vec![],
        Utc::now(),
    )
    .unwrap();

    // Step 1: Open database and write observation under current migration
    {
        let storage = Storage::open(&db_path).unwrap();
        storage.publish_observation(obs.clone()).unwrap();
    }

    // Step 2: Manually run migrations again on the existing connection
    {
        let mut conn = rusqlite::Connection::open(&db_path).unwrap();
        let version = crate::storage::migrations::run_migrations(&mut conn).unwrap();
        assert_eq!(version, crate::storage::CURRENT_SCHEMA_VERSION);
    }

    // Step 3: Reopen through Storage, verify record remains intact
    {
        let storage = Storage::open(&db_path).unwrap();
        let page = storage
            .read_observations(&ObservationFilter::default())
            .unwrap();
        assert_eq!(page.total_count, 1);
        assert_eq!(page.items[0], obs);
    }
}
