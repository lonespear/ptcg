#!/bin/bash
# D47 Phase 1 driver: download a day, build the leaf training table, purge.
# Usage: leaf_days.sh DATE [DATE ...]   (token read from the Keychain at
# use time on the laptop; on a machine with no Keychain entry the episode
# dumps must already be in ~/.cache/kagglehub — pass NODL=1 to skip download)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/bin/python}"
WORKERS="${WORKERS:-8}"
TOKEN=""
if [ -z "${NODL:-}" ]; then
  TOKEN="$(security find-generic-password -a "$USER" -s kaggle-api-token -w 2>/dev/null || true)"
fi
for d in "$@"; do
  if [ -f "data/leaf_train/$d/meta.json" ]; then
    echo "=== $d already built — skipping"
    continue
  fi
  echo "=== $d ==="
  if [ -n "${NODL:-}" ]; then
    "$PY" -m ptcg.leaf.build_table --day "$d" --workers "$WORKERS" --purge
  else
    KAGGLE_API_TOKEN="$TOKEN" "$PY" -m ptcg.leaf.build_table \
      --day "$d" --download --purge --workers "$WORKERS"
  fi
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== $d FAILED rc=$rc — stopping"
    exit $rc
  fi
done
echo "=== all days done"
