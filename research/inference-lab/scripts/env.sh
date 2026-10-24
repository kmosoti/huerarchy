#!/usr/bin/env bash
# Source this file from bash to select the workspace-local tools.
research_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_HOME="$research_root/../scratch/inference-cargo"
export RUSTUP_HOME="$research_root/../scratch/inference-rustup"
export PATH="$research_root/.venv/bin:$CARGO_HOME/bin:$PATH"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
