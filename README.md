# Huerarchy: Foundation Bootstrap v0.1

Huerarchy is a private, Rust-native computational substrate for autonomous AI agents.

This repository implements the first usable foundation milestone: establishing shared local state, deterministic interfaces, and enabling multiple agent runtimes (Claude Code, Codex, Antigravity) to interact with the exact same environment.

## 1. Architectural Invariants

1. **Deterministic Interfaces**: All observations are read in deterministic order (`ORDER BY created_at ASC, id ASC`).
2. **Explicit Bounded Limits**:
   - Subject: max 256 chars.
   - Identifier: max 128 chars (`[a-zA-Z0-9-_:./]`).
   - Statement payload: max 64 KiB.
   - Evidence references: max 16 items per observation.
   - Pagination: strictly bounded between 1 and 100 records per page (default 50).
3. **Idempotency by Design**: Publishing an identical observation with the same ID succeeds idempotently (`already_exists`). Publishing conflicting content under an existing ID is rejected with `DuplicateIdConflict`.
4. **Pure Domain Boundary**: `huerarchy-core` is pure Rust logic with zero dependencies on SQLite, filesystem, network transport, or MCP SDK.
5. **Protocol Isolation**: MCP JSON-RPC protocol messages are restricted exclusively to `stdout`. All diagnostics and tracing are written to `stderr`.
6. **No Speculative Orchestration**: Autonomous agent self-coordination, distributed mesh, and message brokers are explicitly deferred until local shared state is verified.

---

## 2. Implemented Features

- **Domain Core (`huerarchy-core`)**: Pure Rust validated models for `Participant`, `Capability`, `Observation`, and `EvidenceRef`.
- **SQLite WAL Persistence (`huerarchy-server`)**:
  - Embedded SQLite WAL mode (`PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA busy_timeout = 5000;`).
  - Versioned linear schema migrations (`PRAGMA user_version`).
  - Durable observations surviving process restarts.
  - Concurrent multi-threaded write locking with deadlock prevention (`BEGIN IMMEDIATE`).
  - Bounded pagination and deterministic queries with composite indexes.
  - Safe default directory resolution (`~/.local/share/huerarchy/huerarchy.db`) with `0700` directory permissions.
- **Model Context Protocol (MCP) Server**:
  - Stdio transport implementation using official `rmcp` SDK.
  - Four exposed tools:
    1. `health`: Substrate readiness, schema version, storage metrics.
    2. `list_capabilities`: Capability discovery and JSON schema contracts.
    3. `publish_observation`: Input validation, bounded constraints, idempotency enforcement.
    4. `read_observations`: Filtered, paginated, deterministically ordered retrieval.
- **Protocol Test Harness**:
  - In-memory duplex client smoke test verifying all 4 tools.
  - Subprocess stdio test verifying line-delimited JSON-RPC execution.

---

## 3. Workspace Structure

```text
huerarchy/
├── Cargo.toml                  # Workspace root configuration
├── .gitignore                  # Ignores build artifacts, databases, and logs
├── crates/
│   ├── huerarchy-core/         # Pure domain types and invariant checks
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── capability.rs   # Capability contracts
│   │       ├── error.rs        # DomainError enumeration
│   │       ├── evidence.rs     # EvidenceRef and EvidenceKind
│   │       ├── id.rs           # ParticipantId, CapabilityId, ObservationId
│   │       ├── lib.rs          # Re-exports and domain unit tests
│   │       ├── observation.rs  # Observation, ObservationStatement, ObservationFilter
│   │       └── participant.rs  # Participant and ParticipantKind
│   └── huerarchy-server/       # Storage engine, MCP server, and CLI binary
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs          # Library interface
│       │   ├── main.rs         # CLI entrypoint (stdio MCP server & inspect commands)
│       │   ├── mcp/            # MCP server implementation
│       │   │   ├── mod.rs      # rmcp tool handlers and stdio server runner
│       │   │   └── types.rs    # Tool parameter and response schemas
│       │   └── storage/        # SQLite WAL persistence layer
│       │       ├── error.rs    # StorageError enumeration
│       │       ├── migrations.rs # Linear schema migrations
│       │       ├── mod.rs      # Storage handle and queries
│       │       ├── paths.rs    # Database path resolution and permissions
│       │       └── tests.rs    # Storage persistence and concurrency tests
│       └── tests/
│           └── mcp_protocol_test.rs # Protocol-level duplex and stdio smoke tests
└── docs/
    └── adr/
        ├── 0001-sqlite-wal-persistence.md
        ├── 0002-observation-idempotency-policy.md
        └── 0003-agent-interoperability-via-stdio-mcp.md
```

---

## 4. Build and Verification Commands

### Build Release Binary
```bash
cargo build --release
```
The compiled binary will be located at `target/release/huerarchy`.

### Run Test Suite
```bash
cargo test --workspace
```

### Run Format and Clippy Checks
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

---

## 5. Local CLI & MCP Launch Instructions

### Run CLI Inspection Commands
```bash
# Check substrate health and database status
cargo run -p huerarchy-server -- health

# List registered capabilities
cargo run -p huerarchy-server -- capabilities list

# Read observations
cargo run -p huerarchy-server -- observations list --limit 10
```

### Launch MCP Server Manually
```bash
./target/release/huerarchy run
# Or with explicit database path:
./target/release/huerarchy --db-path /path/to/custom.db run
```

---

## 6. Agent Configuration Commands

Connect Claude Code, Codex, and Antigravity to the local Huerarchy MCP server:

### Claude Code (v2.1.296)
```bash
claude mcp add huerarchy -- /home/kmosoti/Projects/moiric/huerarchy/target/release/huerarchy run
```
Or via `.mcp.json` in a project directory:
```json
{
  "mcpServers": {
    "huerarchy": {
      "command": "/home/kmosoti/Projects/moiric/huerarchy/target/release/huerarchy",
      "args": ["run"]
    }
  }
}
```

### Codex CLI (v0.160.0)
```bash
codex mcp add huerarchy -- /home/kmosoti/Projects/moiric/huerarchy/target/release/huerarchy run
```
Or in `~/.codex/config.toml`:
```toml
[mcp_servers.huerarchy]
command = "/home/kmosoti/Projects/moiric/huerarchy/target/release/huerarchy"
args = ["run"]
```

### Antigravity CLI (`agy` 1.3.3)
Add to `~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "huerarchy": {
      "command": "/home/kmosoti/Projects/moiric/huerarchy/target/release/huerarchy",
      "args": ["run"]
    }
  }
}
```

---

## 7. Known Limitations

- **Local Workstation Only**: Operates strictly on local filesystem storage via SQLite WAL.
- **Single Active Substrate Process per Stdio Session**: Each stdio connection spawns a worker process. Multiple processes safely share the SQLite file via WAL locking.
- **Ephemeral in-memory mode**: In-memory mode (`--db-path :memory:`) resets on process exit; default filesystem path should be used for durable persistence.

---

## 8. Explicitly Deferred Systems

The following capabilities are deliberately deferred to future milestones:
- Autonomous multi-agent coordination or goal decomposition.
- Distributed network transport (QUIC, HTTP/SSE, Tailscale mesh communication).
- External capability runtimes (Wasmtime, Z3 theorem proving, OR-Tools).
- Observability ingestion bridges (FabricO11y).
- Cryptographic signature validation of participant identities.
