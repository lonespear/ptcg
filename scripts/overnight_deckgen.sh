#!/bin/bash
# Overnight D30 deck-generation run (Aug 7->8 rotation). One profile per
# machine; launch is a go decision, this script just makes it a switch-flip.
#
#   laptop:    ./scripts/overnight_deckgen.sh laptop   [--resume]
#   sebastian: ./scripts/overnight_deckgen.sh sebastian [--resume]
#
# Both fitness functions are identical (same panel top-8, D18 termination
# penalty, frozen jon pilot rules-policy); only sizing differs. Different
# seeds so the two archipelagos explore different lineages — the morning
# merge is scripts/select_stable.py over both archive.jsonl files.
set -euo pipefail
cd "$(dirname "$0")/.."

MACHINE="${1:?usage: overnight_deckgen.sh laptop|sebastian [--resume]}"
shift || true

case "$MACHINE" in
  laptop)
    RUN_ID=stable_r1
    ARGS=(--hours 9 --pop 10 --games 24 --workers 7 --seed 71)
    PY=/Users/austinsemmel/Desktop/PTCG_AI/.venv/bin/python
    ;;
  sebastian)
    # 10-core Mini sharing the box with the agent session and the standing
    # mining job: fewer workers, smaller pops, fewer games per opponent.
    RUN_ID=stable_r1_seb
    ARGS=(--hours 9 --pop 8 --games 16 --workers 4 --seed 72)
    PY=.venv/bin/python
    ;;
  *) echo "unknown machine: $MACHINE" >&2; exit 1 ;;
esac

mkdir -p runs
"$PY" scripts/run_archipelago.py --run-id "$RUN_ID" --pilot jon \
    "${ARGS[@]}" "$@" 2>&1 | tee -a "runs/driver_${RUN_ID}.log"
