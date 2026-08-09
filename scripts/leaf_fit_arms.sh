#!/bin/bash
# D47 Phase 2 racing arms. Usage: leaf_fit_arms.sh arm [arm ...]
# arm = h<hidden>_<objective>, e.g. h64_multi h128_rank
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/bin/python}"
DATA="${DATA:-data/leaf_train/dataset.npz}"
EPOCHS="${EPOCHS:-10}"
for arm in "$@"; do
  h="${arm#h}"; h="${h%%_*}"
  obj="${arm#*_}"
  echo "=== $arm (hidden $h, objective $obj) ==="
  "$PY" -m ptcg.leaf.train fit --data "$DATA" --hidden "$h" \
      --objective "$obj" --epochs "$EPOCHS" \
      --out "data/leaf_train/model_$arm" 2>&1 | tee "runs/fit_$arm.log"
done
echo "arms done: $*"
