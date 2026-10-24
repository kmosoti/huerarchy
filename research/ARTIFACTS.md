# Research code, results, and full evidence

`inference-lab/` is a standalone research project. The parent Cargo workspace
excludes it so the existing Huerarchy core/server continue to build independently.
The imported experimental sources are byte-for-byte copies of the measured
implementation. The nested `.gitignore` adds exclusions for release-only traces.
The research is not wired into the production MCP tools or storage schema.

## What is saved where

Git contains the complete Rust/Python implementation, tests, pinned dependencies,
configurations, source references, reports, charts, per-run compact records,
concepts, proofs, manifests, and verification receipts. Reports can be viewed
without obtaining more teacher answers. The root aggregate research report also
preserves results from the earlier storage/retrieval and learning studies.

Large SQLite checkpoints, compressed observation/prediction traces, world
fixtures, and archived base data are packaged in the original complete archive
`inference-lab-pilot.tar.gz` (3,086,546,741 bytes). It remains available in the task
workspace at `/workspace/inference-lab-pilot.tar.gz`. It is also split into four
ordered parts in `/workspace/research-release-assets`, below GitHub's per-asset
size limit.

**Remote archive upload is incomplete.** Git pushes and PR creation succeeded,
but GitHub's `uploads.github.com` endpoint rejected the cloud credentials with
HTTP 401. The empty draft release was removed. No full archive or part files are
stored in this Git repository or in its releases. Download the archive from the
task workspace for durable storage elsewhere; a Git clone alone cannot recover
these omitted files. This limitation does not affect the committed code, compact
results, reports, or recorded observations.

[artifact-inventory.json](artifact-inventory.json) records the original archive
SHA-256, every part's SHA-256/size, and the archived files omitted from Git.
Upstream model binaries, build outputs, virtual environments, and large
regenerable embedding matrices were already excluded from the original archive;
the source-pinned setup and fixtures describe their reconstruction.

The saved manifests refer to the full evidence set. Verification, checkpoint
replay, or report regeneration that reads traces requires restoring that set.
A plain Git checkout intentionally does not contain those large files.

## Restore the original evidence

Download `inference-lab-pilot.tar.gz` from the task workspace and place it in
`scratch/research-2026-10-24/` in your checkout. From the repository root:

```bash
mkdir -p scratch/research-2026-10-24
python3 - <<'PY'
import hashlib, json
from pathlib import Path
metadata = json.loads(Path('research/artifact-inventory.json').read_text())['archive']
archive = Path('scratch/research-2026-10-24') / metadata['name']
h = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
assert archive.stat().st_size == metadata['bytes']
assert h.hexdigest() == metadata['sha256']
print('Original archive size and SHA-256 verified')
PY
tar -xzf scratch/research-2026-10-24/inference-lab-pilot.tar.gz \
  --exclude=inference-lab/.gitignore -C research
```

This restores the same experiment results; it does not rerun learning. Extract
into a clean checkout or separate directory if you have local research edits.
Alternatively, download all four workspace part files and `SHA256SUMS`,
concatenate `inference-lab-pilot.tar.gz.part*` into `inference-lab-pilot.tar.gz`,
and run `sha256sum -c SHA256SUMS`. The inventory has the individual part hashes.

## Build and test the research independently

The recorded environment used Python 3.12.14, Rust 1.99.1, a four-core CPU quota,
16 GiB RAM, and no GPU. The active suite does not require a model download.
With `uv` and Rust installed, from the repository root:

```bash
cd research/inference-lab
mkdir -p ../scratch
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock
cargo build --release --locked
cargo test --locked
cargo fmt --check
cargo clippy --all-targets -- -D warnings
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests
```

After restoring evidence, verify the complete latest study and replay a checkpoint:

```bash
.venv/bin/python scripts/check-active.py results/active/default
.venv/bin/python -m research_lab active replay --run results/active/default \
  --id b0/stable/mechanism-0/noise-0/seed-0/epig-change
```

The check refreshes its verification receipt and manifest. A rebuild at a new
path can change the binary hash; the receipt records the actual executable used.
It does not obtain new teaching answers.

For a fresh smoke study, choose an output directory that does not already exist:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m research_lab active run --smoke \
  --output results/active/local-smoke
```

The [research README](inference-lab/README.md) documents the full bounded study,
plugin interfaces, all earlier experiments, and optional native embedding setup.
Its original `/workspace/inference-lab` examples refer to the source workspace;
in this repository use `research/inference-lab` instead. Fresh runs must use new
output paths so the preserved study is not overwritten.
