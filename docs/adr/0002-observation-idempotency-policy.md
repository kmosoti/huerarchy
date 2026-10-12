# ADR 0002: Observation Idempotency Policy

## Context
Autonomous agents frequently retry failed or timed-out tool calls. If a network blip or timeout occurs after an observation was committed, repeating the `publish_observation` call with the same ID must not corrupt state or cause erroneous duplicate entries. Conversely, reusing an ID for contradictory data must be firmly rejected.

## Decision
We enforce an explicit idempotency policy on `publish_observation`:
1. If the observation ID does not exist in the database:
   - Insert the observation and associated evidence transactionally.
   - Return status `created`.
2. If the observation ID already exists:
   - Compare semantic content: `author`, `subject`, `statement` (kind + payload), and `evidence` (list of kind, locator, description).
   - If all semantic fields match:
     - Return the existing observation record with status `already_exists` (idempotent replay success).
   - If any semantic field differs:
     - Abort the transaction and return a structured conflict error (`DuplicateIdConflict: an observation with id '{id}' already exists with different contents`).

## Consequences
- Replaying identical observation publishing requests is safe and idempotent.
- Identifier collisions with conflicting claims are immediately detected and rejected.
- Timestamp comparison is omitted from the idempotency check so retries can report new request timestamps while resolving to the original recorded observation.
