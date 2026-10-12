# Huerarchy: Foundation Bootstrap v0.1

You are the founding implementation engineer for **Huerarchy**, a private, Rust-native computational substrate for autonomous AI agents.

Your assignment is to **implement, test, and document the first usable foundation**, not merely produce a design.

Work directly in the current directory.

## 1. Project mission

Huerarchy will eventually provide a shared computational environment in which Claude Code, Codex, Antigravity, and their subagents can:

- Discover opportunities and capabilities.
- Manufacture reusable computational capabilities.
- Communicate and collaborate without permanent assigned roles.
- Acquire bounded commitments and resources.
- Produce independently verifiable engineering outcomes.
- Preserve knowledge across sessions and machines.

The human defines objectives, permissions, and resource boundaries. Agents eventually determine how to organize within those boundaries.

**Do not implement autonomous coordination yet.**

This milestone establishes shared state, deterministic interfaces, and the ability for different agent runtimes to interact with the same environment.

## 2. Known environment

Development host: Fedora 44, `fedora-blade`.

Installed agent interfaces:

- Claude Code 2.1.296
- Codex CLI 0.160.0
- Antigravity CLI (`agy`) 1.3.3

Claude and Codex authentication have been verified.

The machine has approximately 32 GiB RAM, an NVIDIA RTX 3060 with 6 GiB VRAM, and substantial available disk space.

An existing Tailscale network connects Fedora to two DigitalOcean machines.

Do not modify Tailscale, networking, SSH, firewall rules, GPU configuration, or existing agent credentials.

OVH machines will be provisioned later.

## 3. Engineering principles

Optimize for:

1. Correctness and explicit invariants.
2. Minimal necessary code and dependencies.
3. Computational and memory efficiency.
4. Clear component boundaries.
5. Deterministic behavior where possible.
6. Testable interfaces and reproducible results.
7. Failure isolation and recoverability.
8. Security by default.
9. Evidence rather than unsupported completion claims.

Prefer Rust 1.100-compatible stable language features when the installed toolchain supports them.

Use a functional core with an imperative shell.

Do not introduce speculative abstractions, generic repository frameworks, service locators, elaborate dependency injection, or unnecessary crates.

Avoid YAML configuration.

Favor typed Rust models, algebraic enums, and explicit error handling.

Use established, well-maintained FOSS libraries rather than rebuilding mature primitives.

Before adding a dependency, verify that its maintenance, licensing, and technical requirements are appropriate.

## 4. Repository initialization

Repository name: `huerarchy`

Owner: `kmosoti`

Visibility: **PRIVATE**

Target GitHub repository: `kmosoti/huerarchy`

Inspect the current directory and Git state before changing anything.

If the repository already exists, preserve its history and work with the existing state.

Otherwise initialize a new Rust project and Git repository.

If GitHub CLI authentication and repository permissions are available, create the private repository and push the initial implementation after local verification.

Never create a public repository.

Never overwrite, force-push, or modify an unrelated repository.

If private remote creation is unavailable, complete the local work and report the exact blocker.

## 5. Foundation architecture

Implement a minimal Rust workspace with these logical boundaries:

### Domain core

Pure Rust types and deterministic state-transition logic.

Initial domain concepts:

- Participant
- Capability
- Observation
- Evidence reference

Prefer compact structs and enums.

Domain logic must not depend directly on SQLite, the filesystem, network transport, or the MCP runtime.

### Persistence

Use SQLite WAL through a maintained Rust library such as `rusqlite`.

Implement:

- Versioned database migrations.
- Durable observations.
- Capability declarations.
- Stable identifiers.
- Timestamps and provenance.
- Transactional writes.
- Busy-timeout handling.
- Deterministic ordering for reads.
- Appropriate indexes.

Keep the database on a local filesystem.

Configure its location explicitly, with a safe per-user development default.

Do not persist credentials or model conversation transcripts.

### MCP interface

Implement a real MCP server using an appropriate maintained Rust SDK.

**Initial transport: stdio.**

Do not implement HTTP, QUIC, custom TCP protocols, or distributed networking yet.

Expose four tools:

1. `health`
   Returns service readiness, protocol/schema version, and storage status.

2. `list_capabilities`
   Returns registered capabilities and their documented contracts.

3. `publish_observation`
   Records a typed observation with its author, subject, statement, and provenance reference. Validate inputs and enforce bounds.

4. `read_observations`
   Retrieves observations using bounded pagination and optional filters, in deterministic order.

All tool results should be structured and machine-readable.

Keep MCP protocol output isolated on stdout. Diagnostics belong on stderr.

## 6. Implementation constraints

Start with the smallest workspace organization that preserves the domain boundary. Two crates are acceptable; eight speculative crates are not.

Prefer:

- `serde` for serialization.
- `rusqlite` for SQLite.
- `rmcp` if its current API and maintenance meet the requirements.
- `tracing` for structured diagnostics.
- `thiserror` or another small error mechanism where justified.

Do not introduce a message broker, graph database, vector database, scheduler, workflow engine, or distributed consensus.

Do not implement Wasmtime, Z3, OR-Tools, or FabricO11y integration in this milestone.

Those are future capability adapters.

FabricO11y is Huerarchy's selected observability platform. Preserve an appropriate interface boundary for future instrumentation, but do not fabricate ingestion compatibility or require FabricO11y to run.

## 7. Testing and verification

Implement tests that demonstrate:

- Domain validation rejects malformed input.
- SQLite persists observations across process restarts.
- Concurrent writes preserve database consistency.
- Observation retrieval is stable and bounded.
- Duplicate submissions behave according to an explicit idempotency policy.
- Database migrations preserve existing data.
- MCP tool schemas and responses match the advertised interface.
- Unexpected failures return structured errors rather than false success.

At minimum run:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`

If feasible, add a protocol-level MCP smoke test that invokes the server through stdio and exercises the four tools.

Use a temporary test database. Preserve actual failures and do not weaken tests to manufacture success.

Bound test execution time and avoid excessive local CPU/memory consumption.

## 8. Agent compatibility

Inspect the available MCP configuration capabilities of:

- Claude Code
- Codex
- Antigravity

Document exact, version-compatible commands for connecting each agent to the local Huerarchy MCP server.

Do not modify global agent configuration automatically.

Do not run paid model inference to validate integration.

Prefer a standalone MCP client or protocol test harness.

All three agents must eventually be capable of reading the same observation through the same interface.

## 9. Security boundaries

This milestone is local-only.

- Never expose the MCP endpoint to a public network.
- Never use permission-bypass flags.
- Never read or publish authentication secrets.
- Never give the MCP server arbitrary shell execution capabilities.
- Never allow user-provided paths to escape authorized storage boundaries.
- Never treat an observation as an executable instruction.
- Limit input sizes, result sizes, and pagination.
- Keep SQLite and generated artifacts outside tracked source files.
- Make the data format and migration policy explicit.

Later versions will add authenticated inter-node communication over the existing private infrastructure.

Do not implement it now.

## 10. Required documentation

Create a concise README containing:

- Project mission and architectural invariants.
- Actual implemented features.
- Workspace structure.
- Build and test commands.
- Local MCP launch instructions.
- SQLite schema and migration approach.
- Agent connection commands.
- Known limitations.
- Explicitly deferred features.

Document significant architectural decisions in lightweight ADRs only when they establish real constraints.

Do not generate pages of aspirational documentation.

## 11. Work method

Follow this sequence:

1. Inspect the repository, Rust toolchain, and installed local tools.
2. Establish a minimal implementation plan.
3. Implement the domain and persistence layer.
4. Test persistence and invariants.
5. Implement the MCP adapter.
6. Execute a protocol-level smoke test.
7. Run formatting, Clippy, and workspace tests.
8. Review the resulting design for duplication and unnecessary code.
9. Remove unnecessary abstractions.
10. Commit the verified foundation.
11. Create/push the private remote when authorized access permits.

Do not stop after planning.

Do not recursively spawn other coding agents for this bootstrap.

Make changes incrementally and validate each meaningful slice.

Limit repeated attempts to fix the same failure. After two unsuccessful approaches, identify the underlying cause and change strategy rather than cycling indefinitely.

Never claim tests passed unless they actually executed successfully.

## 12. Acceptance criteria

The milestone is complete when:

- The Rust workspace builds.
- SQLite-backed observations survive restart.
- The MCP server launches locally.
- A protocol client can discover and invoke all four tools.
- Published observations are retrievable.
- Tests validate the documented invariants.
- The source contains no unnecessary orchestration infrastructure.
- Local agent configuration instructions are documented.
- Repository visibility is private if remote creation succeeded.

## 13. Final output

Return a concise human-readable summary followed by one valid JSON object containing:

- `repository`: local path, remote URL, and confirmed visibility.
- `toolchain`: Rust and Cargo versions.
- `implementation`: crates, interfaces, and schema version.
- `verification`: exact command, exit code, and result for each check.
- `mcp`: smoke-test results and supported transports.
- `agents`: connection readiness for Claude, Codex, and Antigravity.
- `security`: confirmed boundaries and unresolved risks.
- `deferred`: deliberately excluded systems.
- `blockers`: anything not completed.
- `next_step`: the smallest useful follow-up.

Use `null` for unavailable facts rather than guessing.

Do not print credentials, tokens, or private configuration contents.

**Begin by inspecting the environment, then implement the foundation.**

