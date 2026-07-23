#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────
# LoRC — Hyperparameter Sweep Launcher
#
# Delegates to `lore/run_sweep.py` which reads
# `lore/sweep_config.yaml` and runs the full Cartesian grid.
#
# Usage:
#   ./sweep.sh                           # full grid (sequential, GPU)
#   ./sweep.sh --parallel 4              # 4 concurrent runs
#   ./sweep.sh --dry-run                 # inspect combinations
#   ./sweep.sh --force                   # rerun completed
#   ./sweep.sh --config lore/sweep_config_custom.yaml
#   CUDA_VISIBLE_DEVICES=0,1 ./sweep.sh --parallel 2  # multi-GPU
# ─────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT"

exec python3 -m lorc.run_sweep "$@"
