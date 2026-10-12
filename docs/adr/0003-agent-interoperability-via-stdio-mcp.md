# ADR 0003: Agent Interoperability via Stdio Model Context Protocol

## Context
Claude Code, Codex CLI, and Antigravity CLI run on the developer's workstation. To provide a shared computational substrate across all three runtime environments, Huerarchy must expose a standard protocol interface without opening network ports, requiring remote credentials, or allowing unauthorized access.

## Decision
We implement a Model Context Protocol (MCP) server using the official Rust `rmcp` SDK over **stdio transport**.
- **Stdio Transport**: Subprocess stdin/stdout streams carry standard JSON-RPC 2.0 messages.
- **Protocol Isolation**: MCP JSON-RPC protocol messages are restricted exclusively to `stdout`. All tracing, diagnostics, and error logs are directed to `stderr`.
- **Tool Surface**:
  1. `health`: readiness, protocol version, schema version, storage stats.
  2. `list_capabilities`: capability registry and machine-readable contracts.
  3. `publish_observation`: validated observation publishing.
  4. `read_observations`: deterministically ordered, bounded paginated observation queries.
- **Dual Content Compatibility**: Tool responses populate both `content` (formatted JSON text for older MCP clients) and `structured_content` (parsed JSON values for modern structured-content MCP clients).

## Consequences
- Zero open TCP ports or network listeners.
- Claude Code, Codex, and Antigravity can concurrently spawn or connect to the Huerarchy binary.
- All agents read and write the exact same shared state in SQLite.
- Distributed networking is deferred to future private-mesh milestones.
