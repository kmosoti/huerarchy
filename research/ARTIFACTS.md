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
fixtures, and archived base data are saved in the private draft release tagged
`research-2026-10-24`. The original complete archive is split into ordered parts
below GitHub's per-asset size limit. This avoids committing gigabytes of generated
data to Git history. A draft release has not been announced as a software release.

[artifact-inventory.json](artifact-inventory.json) records the original archive
SHA-256, every part's SHA-256/size, and the archived files omitted from Git.
Upstream model binaries, build outputs, virtual environments, and large
regenerable embedding matrices were already excluded from the original archive;
the source-pinned setup and fixtures describe their reconstruction.

The saved manifests refer to the full evidence set. Verification, checkpoint
replay, or report regeneration that reads traces requires restoring that set.
A plain Git checkout intentionally does not contain those large files.

## Restore the original evidence

Run from the repository root using GitHub CLI authenticated for this private repo:

```bash
mkdir -p scratch/research-2026-10-24
gh release download research-2026-10-24 --repo kmosoti/huerarchy \
  --dir scratch/research-2026-10-24
cd scratch/research-2026-10-24
cat inference-lab-pilot.tar.gz.part* > inference-lab-pilot.tar.gz
sha256sum -c SHA256SUMS
cd ../..
tar -xzf scratch/research-2026-10-24/inference-lab-pilot.tar.gz \
  --exclude=inference-lab/.gitignore -C research
```

This restores the same experiment results; it does not rerun learning. Extract
into a clean checkout or separate directory if you have local research edits.
Asset access requires repository access. All four part files are necessary.

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
