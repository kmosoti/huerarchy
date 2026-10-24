#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$research_root"
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release
python -m unittest discover -s tests -v
python -m research_lab discovery run --config configs/discovery/default.json "$@"
