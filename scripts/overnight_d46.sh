#!/bin/bash
# D46 large-scale overnight run — seeded archipelago v4, six chains split
# across two machines (panel: data/panel_ladder_v3.json, weighted).
#
#   laptop:    ./scripts/overnight_d46.sh laptop    [--resume] [extra args]
#   sebastian: ./scripts/overnight_d46.sh sebastian [--resume] [extra args]
#
# Chain split (documented in runs/seeded_d46/README.md): archaludon runs on
# BOTH machines under different RNG seeds; migration is intra-process, so
# each machine hosts full explore+refine pairs for its chains.
set -euo pipefail
cd "$(dirname "$0")/.."

MACHINE="${1:?usage: overnight_d46.sh laptop|sebastian [--resume]}"
shift || true

case "$MACHINE" in
  laptop)
    CHAINS=grimmsnarl-mirror,archaludon,rainbow-Kanga
    WORKERS=7; SEED=81
    PY=/Users/austinsemmel/miniforge3/bin/python3.10
    ;;
  sebastian)
    CHAINS=spec-Ogerpon,engine,counter-900,archaludon
    WORKERS=10; SEED=82
    PY=/Users/sebastian/ptcg_fork/.venv/bin/python
    ;;
  *) echo "unknown machine: $MACHINE" >&2; exit 1 ;;
esac

mkdir -p runs
"$PY" scripts/run_seeded.py --run-id seeded_d46 \
    --panel data/panel_ladder_v3.json \
    --chains "$CHAINS" --workers "$WORKERS" --seed "$SEED" \
    --explore-pop 36 --refine-pop 14 \
    --plateau 24 --improve-eps 0.005 \
    --migrate-every 10 --migrate-top 2 --floor-wr 0.35 \
    --deep-block 50 --deep-max 400 \
    --founders runs/seeded_d46/founders.json \
    "$@" 2>&1 | tee -a "runs/driver_d46_${MACHINE}.log"
