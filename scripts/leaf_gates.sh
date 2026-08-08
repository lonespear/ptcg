#!/bin/bash
# D47 pre-registered gates: candidate (neural leaf) vs baseline (shipped v6),
# paired seeds 94000+, 300 games/arm. Usage:
#   leaf_gates.sh g1          # 4-cell gatekeeper, our proven list
#   leaf_gates.sh g2          # driving cells (v6b decks)
# Env: CAND (default build/agents/neural_leaf_v1/main.py), GAMES, WORKERS, OUT
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniforge3/bin/python}"
CAND="${CAND:-build/agents/neural_leaf_v1/main.py}"
GAMES="${GAMES:-300}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-94000}"
OUT="${OUT:-$HOME/Desktop/PTCG_AI/experiments_0808}"
mkdir -p "$OUT"

run_cell() {  # name deck_a spec_agent spec_deck arm main
  local name="$1" deck_a="$2" sagent="$3" sdeck="$4" arm="$5" main="$6"
  local out="$OUT/d47_gate_${name}_${arm}.json"
  if [ -f "$out" ]; then echo "skip $name/$arm (exists)"; return; fi
  echo "=== $name / $arm ==="
  "$PY" scripts/specialist_gate.py --a "$main" --deck-a "$deck_a" \
      --specialist "$sagent" --deck-specialist "$sdeck" \
      --games "$GAMES" --workers "$WORKERS" --seed "$SEED" --out "$out"
}

case "${1:-all}" in
  g1|all)
    for arm_main in "cand:$CAND" "base:agent/main.py"; do
      arm="${arm_main%%:*}"; main="${arm_main#*:}"
      run_cell grimmsnarl agent/deck.csv external/grimmsnarl_agent.py \
          external/grimmsnarl_deck.json "$arm" "$main"
      run_cell alakazam agent/deck.csv external/codex_alakazam_agent.py \
          external/codex_alakazam_deck.json "$arm" "$main"
      run_cell archaludon agent/deck.csv external/archaludon_agent.py \
          external/archaludon_deck.json "$arm" "$main"
      run_cell lucario agent/deck.csv external/lucario_agent.py \
          external/lucario_deck.json "$arm" "$main"
    done
    ;;&
  g2|all)
    for arm_main in "cand:$CAND" "base:agent/main.py"; do
      arm="${arm_main%%:*}"; main="${arm_main#*:}"
      run_cell drive_grimm data/v6b_grimmsnarl_deck.csv \
          external/grimmsnarl_agent.py external/grimmsnarl_deck.json \
          "$arm" "$main"
      run_cell drive_engine data/v6b_engine_deck.csv \
          external/archaludon_agent.py external/archaludon_deck.json \
          "$arm" "$main"
    done
    ;;
esac
echo "gates pass ${1:-all} done"
