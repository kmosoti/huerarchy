use huerarchy_server::{HuerarchyServer, Storage};
use rmcp::{model::CallToolRequestParams, ServiceExt};
use tempfile::NamedTempFile;

#[tokio::test]
async fn test_mcp_protocol_in_memory_duplex_all_tools() -> Result<(), Box<dyn std::error::Error>> {
    let temp_file = NamedTempFile::new()?;
    let db_path = temp_file.path().to_path_buf();
    let storage = Storage::open(&db_path)?;
    let server = HuerarchyServer::new(storage);

    let (server_transport, client_transport) = tokio::io::duplex(8192);

    let _server_handle = tokio::spawn(async move {
        if let Ok(service) = server.serve(server_transport).await {
            let _ = service.waiting().await;
        }
    });

    let client = ().serve(client_transport).await?;

    // 1. Tool discovery: list_tools
    let tools_list = client.list_tools(None).await?;
    let tool_names: Vec<String> = tools_list
        .tools
        .iter()
        .map(|t| t.name.to_string())
        .collect();
    assert!(tool_names.contains(&"health".to_string()));
    assert!(tool_names.contains(&"list_capabilities".to_string()));
    assert!(tool_names.contains(&"publish_observation".to_string()));
    assert!(tool_names.contains(&"read_observations".to_string()));

    // 2. Invoke health tool
    let health_call = client
        .call_tool(CallToolRequestParams::new("health"))
        .await?;
    assert_eq!(health_call.is_error, Some(false));
    let structured_health = health_call.structured_content.expect("structured content");
    assert_eq!(structured_health["status"], "ready");
    assert_eq!(structured_health["schema_version"], 1);
    assert_eq!(structured_health["storage"]["observation_count"], 0);

    // 3. Invoke list_capabilities tool
    let caps_call = client
        .call_tool(CallToolRequestParams::new("list_capabilities"))
        .await?;
    assert_eq!(caps_call.is_error, Some(false));
    let structured_caps = caps_call.structured_content.expect("structured content");
    let caps_array = structured_caps["capabilities"]
        .as_array()
        .expect("array of capabilities");
    assert!(caps_array.len() >= 4);

    // 4. Invoke publish_observation tool
    let obs_payload = serde_json::json!({
        "author": "agent:claude-code",
        "subject": "repo:huerarchy",
        "statement": {
            "kind": "invariant_check",
            "payload": {
                "invariants_verified": 9,
                "status": "passed"
            }
        },
        "evidence": [
            {
                "kind": "git_commit",
                "locator": "b940084d72026092800000000000000000000000",
                "description": "verified commit hash"
            }
        ],
        "id": "obs-protocol-test-01"
    });

    let publish_call = client
        .call_tool(
            CallToolRequestParams::new("publish_observation")
                .with_arguments(obs_payload.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(publish_call.is_error, Some(false));
    let publish_res = publish_call.structured_content.expect("structured content");
    assert_eq!(publish_res["status"], "created");
    assert_eq!(publish_res["observation"]["id"], "obs-protocol-test-01");
    assert_eq!(publish_res["observation"]["author"], "agent:claude-code");

    // 5. Test idempotency: republish exact same observation
    let republish_call = client
        .call_tool(
            CallToolRequestParams::new("publish_observation")
                .with_arguments(obs_payload.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(republish_call.is_error, Some(false));
    let republish_res = republish_call
        .structured_content
        .expect("structured content");
    assert_eq!(republish_res["status"], "already_exists");
    assert_eq!(republish_res["observation"]["id"], "obs-protocol-test-01");

    // 6. Test conflict: publish same ID with conflicting statement
    let conflicting_payload = serde_json::json!({
        "author": "agent:claude-code",
        "subject": "repo:huerarchy",
        "statement": {
            "kind": "invariant_check",
            "payload": {
                "invariants_verified": 999
            }
        },
        "id": "obs-protocol-test-01"
    });
    let conflict_call = client
        .call_tool(
            CallToolRequestParams::new("publish_observation")
                .with_arguments(conflicting_payload.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(conflict_call.is_error, Some(true));

    // 7. Test domain validation failure: malformed author
    let malformed_payload = serde_json::json!({
        "author": "invalid author with spaces!",
        "subject": "repo:huerarchy",
        "statement": {
            "kind": "test",
            "payload": {}
        }
    });
    let malformed_call = client
        .call_tool(
            CallToolRequestParams::new("publish_observation")
                .with_arguments(malformed_payload.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(malformed_call.is_error, Some(true));

    // 8. Invoke read_observations tool
    let read_payload = serde_json::json!({
        "author": "agent:claude-code",
        "limit": 10
    });
    let read_call = client
        .call_tool(
            CallToolRequestParams::new("read_observations")
                .with_arguments(read_payload.as_object().unwrap().clone()),
        )
        .await?;
    assert_eq!(read_call.is_error, Some(false));
    let read_res = read_call.structured_content.expect("structured content");
    assert_eq!(read_res["total_count"], 1);
    let items = read_res["items"].as_array().expect("items array");
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["id"], "obs-protocol-test-01");

    // 9. Verify storage health reflects the new observation
    let final_health = client
        .call_tool(CallToolRequestParams::new("health"))
        .await?;
    let health_data = final_health.structured_content.unwrap();
    assert_eq!(health_data["storage"]["observation_count"], 1);

    Ok(())
}

#[tokio::test]
async fn test_mcp_protocol_stdio_child_process() -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Stdio;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let bin_path = env!("CARGO_BIN_EXE_huerarchy");
    let temp_file = NamedTempFile::new()?;
    let db_path = temp_file.path().to_str().unwrap().to_string();

    let mut child = tokio::process::Command::new(bin_path)
        .args(["--db-path", &db_path, "run"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().expect("child stdin");
    let stdout = child.stdout.take().expect("child stdout");
    let mut reader = BufReader::new(stdout).lines();

    // 1. Send initialize
    let init_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2026-07-28",
            "capabilities": {},
            "clientInfo": { "name": "stdio-smoke-tester", "version": "0.1.0" }
        }
    });
    stdin
        .write_all(format!("{}\n", serde_json::to_string(&init_req)?).as_bytes())
        .await?;
    stdin.flush().await?;

    let init_resp_line = reader.next_line().await?.expect("init response line");
    let init_resp: serde_json::Value = serde_json::from_str(&init_resp_line)?;
    assert_eq!(init_resp["id"], 1);
    assert!(init_resp["result"].is_object());

    // 2. Send initialized notification
    let initialized_notif = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    stdin
        .write_all(format!("{}\n", serde_json::to_string(&initialized_notif)?).as_bytes())
        .await?;
    stdin.flush().await?;

    // 3. Send tools/list
    let list_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });
    stdin
        .write_all(format!("{}\n", serde_json::to_string(&list_req)?).as_bytes())
        .await?;
    stdin.flush().await?;

    let list_resp_line = reader.next_line().await?.expect("tools/list response line");
    let list_resp: serde_json::Value = serde_json::from_str(&list_resp_line)?;
    assert_eq!(list_resp["id"], 2);
    let tools = list_resp["result"]["tools"]
        .as_array()
        .expect("tools array");
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"health"));
    assert!(names.contains(&"list_capabilities"));
    assert!(names.contains(&"publish_observation"));
    assert!(names.contains(&"read_observations"));

    // 4. Call health tool
    let health_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "health",
            "arguments": {}
        }
    });
    stdin
        .write_all(format!("{}\n", serde_json::to_string(&health_req)?).as_bytes())
        .await?;
    stdin.flush().await?;

    let health_resp_line = reader.next_line().await?.expect("health response line");
    let health_resp: serde_json::Value = serde_json::from_str(&health_resp_line)?;
    assert_eq!(health_resp["id"], 3);
    assert_eq!(health_resp["result"]["isError"], false);

    // 5. Call publish_observation tool
    let publish_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "publish_observation",
            "arguments": {
                "author": "agent:stdio-smoke",
                "subject": "e2e:stdio",
                "statement": {
                    "kind": "stdio_verification",
                    "payload": { "roundtrip": true }
                },
                "evidence": [
                    {
                        "kind": "uri",
                        "locator": "file:///test/stdio.log",
                        "description": "stdio log trace"
                    }
                ]
            }
        }
    });
    stdin
        .write_all(format!("{}\n", serde_json::to_string(&publish_req)?).as_bytes())
        .await?;
    stdin.flush().await?;

    let publish_resp_line = reader.next_line().await?.expect("publish response line");
    let publish_resp: serde_json::Value = serde_json::from_str(&publish_resp_line)?;
    assert_eq!(publish_resp["id"], 4);
    assert_eq!(publish_resp["result"]["isError"], false);

    // 6. Call read_observations tool
    let read_req = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "read_observations",
            "arguments": {
                "author": "agent:stdio-smoke"
            }
        }
    });
    stdin
        .write_all(format!("{}\n", serde_json::to_string(&read_req)?).as_bytes())
        .await?;
    stdin.flush().await?;

    let read_resp_line = reader.next_line().await?.expect("read response line");
    let read_resp: serde_json::Value = serde_json::from_str(&read_resp_line)?;
    assert_eq!(read_resp["id"], 5);
    assert_eq!(read_resp["result"]["isError"], false);
    assert_eq!(read_resp["result"]["structuredContent"]["total_count"], 1);

    // Close stdin and terminate child
    drop(stdin);
    let _ = child.kill().await;

    Ok(())
}
