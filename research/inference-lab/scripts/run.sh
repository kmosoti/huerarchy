#!/usr/bin/env bash
set -euo pipefail
research_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$research_root"
source scripts/env.sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
python -m unittest discover -s tests -p 'test_*.py' -v
cargo build --release
python -m research_lab profile
python -m research_lab generate
python -m research_lab audit
python -m research_lab bench --variant all --repeats 5
python -m research_lab report
