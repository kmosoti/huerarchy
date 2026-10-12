# ADR 0001: SQLite WAL Persistence and Migration Strategy

## Context
Huerarchy requires durable, local state persistence that can survive process restarts, tolerate concurrent access from multiple processes/agents, and maintain strict consistency without external database services.

## Decision
We use SQLite compiled with bundled C libraries via `rusqlite`, configured with:
- **Write-Ahead Logging (WAL)** (`PRAGMA journal_mode = WAL;`): Enables concurrent readers alongside a writer.
- **Normal Synchronous** (`PRAGMA synchronous = NORMAL;`): Balances durability and write latency in WAL mode.
- **Foreign Key Constraints** (`PRAGMA foreign_keys = ON;`): Enforces referential integrity (e.g., cascade deletion on evidence references).
- **Busy Timeout** (`conn.busy_timeout(5000ms)`): Prevents immediate `SQLITE_BUSY` contention errors during concurrent write attempts by waiting up to 5 seconds.
- **Immediate Write Transactions** (`BEGIN IMMEDIATE` via `TransactionBehavior::Immediate`): Eliminates deadlocks by acquiring the SQLite reserved lock up-front before write execution.
- **Pragma `user_version` Migrations**: Tracks schema migrations linearly through transactions recorded in `_huerarchy_migrations`.

## Consequences
- Single-node embedded storage requires zero infrastructure setup.
- Database location defaults to secure per-user directory `~/.local/share/huerarchy/huerarchy.db` (mode 0700 on Unix) with `HUERARCHY_DB_PATH` override.
- Network clustering or distributed consensus is deliberately deferred to future milestones.
