pub mod types;

use chrono::Utc;
use huerarchy_core::{Observation, ObservationId, ObservationStatement, ParticipantId};
use rmcp::{
    handler::server::router::tool::ToolRouter, handler::server::wrapper::Parameters,
    model::CallToolResult, tool, tool_handler, tool_router, ServerHandler, ServiceExt,
};
use serde::Serialize;
use tracing::{error, info};

use crate::storage::Storage;
use types::{
    ErrorPayload, HealthResponse, ListCapabilitiesResponse, PublishObservationParams,
    PublishObservationResponse, ReadObservationsParams, ReadObservationsResponse,
};

/// MCP server instance exposing Huerarchy capabilities and observations over stdio.
#[derive(Clone)]
pub struct HuerarchyServer {
    storage: Storage,
    tool_router: ToolRouter<Self>,
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for HuerarchyServer {}

impl HuerarchyServer {
    pub fn new(storage: Storage) -> Self {
        Self {
            storage,
            tool_router: Self::tool_router(),
        }
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Runs the MCP server listening on stdio transport until shutdown.
    pub async fn run_stdio(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting Huerarchy MCP server on stdio transport");
        let (stdin, stdout) = rmcp::transport::stdio();
        let service = self.serve((stdin, stdout)).await?;
        service.waiting().await?;
        info!("Huerarchy MCP server stdio session terminated");
        Ok(())
    }
}

#[tool_router(router = tool_router)]
impl HuerarchyServer {
    /// Returns service readiness, protocol/schema version, and storage status.
    #[tool(
        name = "health",
        description = "Returns substrate readiness, protocol version, database schema version, and storage status."
    )]
    pub async fn health(&self) -> CallToolResult {
        match self.storage.health() {
            Ok(storage_health) => {
                let response = HealthResponse {
                    status: "ready".to_string(),
                    protocol_version: "2026-07-28".to_string(),
                    schema_version: storage_health.schema_version,
                    storage: storage_health,
                };
                make_success_result(&response)
            }
            Err(err) => {
                error!("Health check failed: {err}");
                make_error_result("Health check failed".to_string(), Some(err.to_string()))
            }
        }
    }

    /// Returns registered capabilities and their documented contracts.
    #[tool(
        name = "list_capabilities",
        description = "Returns all registered capabilities and their machine-readable schemas in deterministic order."
    )]
    pub async fn list_capabilities(&self) -> CallToolResult {
        match self.storage.list_capabilities() {
            Ok(capabilities) => {
                let response = ListCapabilitiesResponse { capabilities };
                make_success_result(&response)
            }
            Err(err) => {
                error!("list_capabilities failed: {err}");
                make_error_result(
                    "Failed to list capabilities".to_string(),
                    Some(err.to_string()),
                )
            }
        }
    }

    /// Records a typed observation with its author, subject, statement, and provenance reference. Validate inputs and enforce bounds.
    #[tool(
        name = "publish_observation",
        description = "Records a typed observation with author, subject, statement payload, and provenance reference. Validates inputs and enforces explicit bounds."
    )]
    pub async fn publish_observation(
        &self,
        params: Parameters<PublishObservationParams>,
    ) -> CallToolResult {
        let input = params.0;

        // Domain validation
        let author = match ParticipantId::new(input.author) {
            Ok(a) => a,
            Err(err) => {
                return make_error_result(
                    "Invalid author identifier".to_string(),
                    Some(err.to_string()),
                );
            }
        };

        let obs_id = match input.id {
            Some(id_str) => match ObservationId::new(id_str) {
                Ok(id) => id,
                Err(err) => {
                    return make_error_result(
                        "Invalid observation ID".to_string(),
                        Some(err.to_string()),
                    );
                }
            },
            None => ObservationId::generate(),
        };

        let statement =
            match ObservationStatement::new(input.statement.kind, input.statement.payload) {
                Ok(stmt) => stmt,
                Err(err) => {
                    return make_error_result(
                        "Invalid observation statement".to_string(),
                        Some(err.to_string()),
                    );
                }
            };

        let mut evidence = Vec::new();
        for ev_input in input.evidence {
            match ev_input.into_evidence_ref() {
                Ok(ev) => evidence.push(ev),
                Err(err) => {
                    return make_error_result(
                        "Invalid evidence reference".to_string(),
                        Some(err.to_string()),
                    );
                }
            }
        }

        let observation = match Observation::new(
            obs_id,
            author,
            input.subject,
            statement,
            evidence,
            Utc::now(),
        ) {
            Ok(obs) => obs,
            Err(err) => {
                return make_error_result(
                    "Observation invariant violation".to_string(),
                    Some(err.to_string()),
                );
            }
        };

        // Storage persist with idempotency
        match self.storage.publish_observation(observation) {
            Ok(pub_result) => {
                let response = PublishObservationResponse {
                    status: pub_result.status,
                    observation: pub_result.observation,
                };
                make_success_result(&response)
            }
            Err(err) => {
                error!("Failed to persist observation: {err}");
                make_error_result(
                    "Storage publication error".to_string(),
                    Some(err.to_string()),
                )
            }
        }
    }

    /// Retrieves observations using bounded pagination and optional filters, in deterministic order.
    #[tool(
        name = "read_observations",
        description = "Retrieves durable observations using bounded pagination and optional filters (author, subject, since) in deterministic order."
    )]
    pub async fn read_observations(
        &self,
        params: Parameters<ReadObservationsParams>,
    ) -> CallToolResult {
        let filter = match params.0.into_filter() {
            Ok(f) => f,
            Err(err) => {
                return make_error_result(
                    "Invalid observation filter".to_string(),
                    Some(err.to_string()),
                );
            }
        };

        match self.storage.read_observations(&filter) {
            Ok(page) => {
                let response = ReadObservationsResponse {
                    items: page.items,
                    total_count: page.total_count,
                    limit: page.limit,
                    offset: page.offset,
                    has_more: page.has_more,
                };
                make_success_result(&response)
            }
            Err(err) => {
                error!("Failed to read observations: {err}");
                make_error_result("Storage query error".to_string(), Some(err.to_string()))
            }
        }
    }
}

fn make_success_result<T: Serialize>(value: &T) -> CallToolResult {
    let json_val = serde_json::to_value(value).unwrap_or(serde_json::Value::Null);
    CallToolResult::structured(json_val)
}

fn make_error_result(err_msg: String, details: Option<String>) -> CallToolResult {
    let payload = ErrorPayload {
        error: err_msg,
        details,
    };
    let json_val = serde_json::to_value(&payload).unwrap_or(serde_json::Value::Null);
    CallToolResult::structured_error(json_val)
}
