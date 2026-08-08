#!/bin/bash
# D47 — fresh specialist corpora for the thin cells, all arms in parallel.
# Run from the repo root once the day-extraction run has released the CPUs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venvs/leaf/bin/python}"
mkdir -p runs
"$PY" scripts/leaf_selfplay.py archaludon 600 97000 nat \
    > runs/sp_archaludon_nat.log 2>&1 &
"$PY" scripts/leaf_selfplay.py archaludon 600 98000 gate data/v6b_engine_deck.csv \
    > runs/sp_archaludon_gate.log 2>&1 &
"$PY" scripts/leaf_selfplay.py codex_alakazam 600 97000 nat \
    > runs/sp_alakazam_nat.log 2>&1 &
"$PY" scripts/leaf_selfplay.py grimmsnarl 600 98000 gate data/v6b_grimmsnarl_deck.csv \
    > runs/sp_grimmsnarl_gate.log 2>&1 &
"$PY" scripts/leaf_selfplay.py grimmsnarl 300 97000 nat \
    > runs/sp_grimmsnarl_nat.log 2>&1 &
"$PY" scripts/leaf_selfplay.py lucario 300 97000 nat \
    > runs/sp_lucario_nat.log 2>&1 &
wait
echo "all selfplay arms done"
ls -la data/leaf_train/selfplay_*.jsonl.gz
