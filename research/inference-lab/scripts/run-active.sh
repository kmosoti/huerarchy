#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
cd "$research_root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
python -m unittest discover -s tests
python -m research_lab active run --config configs/active/default.json "$@"
