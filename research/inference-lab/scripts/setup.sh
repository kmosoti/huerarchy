#!/usr/bin/env bash
set -euo pipefail
research_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$research_root"
uv venv --allow-existing --python "${RESEARCH_PYTHON:-python3}" .venv
uv pip sync --python .venv/bin/python requirements.lock
mkdir -p ../scratch results models
if [[ ! -x ../scratch/inference-cargo/bin/rustup ]]; then
    curl --fail --location --silent --show-error https://sh.rustup.rs --output ../scratch/inference-rustup.sh
    CARGO_HOME="$research_root/../scratch/inference-cargo" RUSTUP_HOME="$research_root/../scratch/inference-rustup" \
        sh ../scratch/inference-rustup.sh -y --profile minimal --default-toolchain 1.99.1 --no-modify-path
fi
source scripts/env.sh
rustup component add rustfmt clippy
cargo build --release
python scripts/build_runtime.py
python -m research_lab profile
