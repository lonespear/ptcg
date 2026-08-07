# PLAYBOOK — the agent's strategy, as an auditable table

*Every strategic behavior modifier in the agent is an entry here, or it is not
strategy. Each entry carries three mandatory fields: the **measurement** that
learned it, the **behavior** it drives, and the **gate** it passed before
shipping. Entries a judge can recompute. (Austin's ledger: D23, D25.)*

| # | Entry | Learned from (measurement) | Drives (behavior) | Gate result | Status |
|---|---|---|---|---|---|
| 1 | Threat-curve position pricing: the value of the position we hand over is priced by the posterior-mixed archetype damage curve applied to our board, as a named linear term | D13 goldfish curves (per-archetype damage/turn with quantile bands); posterior graded 0.63@t1→0.94@t10 | `_evaluate` gains a named `incoming_threat` term | pending | BUILDING |
| 2 | Energy-potential ladder: max damage threatenable at t+1/t+2/t+3 for both sides, hypergeometric-weighted by draw odds | Board arithmetic + `outs.py` (10/10 self-checks); targets the measured mid-game blindness (58% of positions in one 0.52 bin, ECE 0.008 calibration) | six named trajectory features in `_evaluate` | pending | BUILDING |
| 3 | Aggression posture: crossover turn between our live ladder and their posterior curve selects RACE / SETUP / WALL / CHIP / DENY, applying a named weight delta | curves + posterior (both measured); posture triggers logged per turn | plan-module weight deltas | pending | R3 |
| 4 | Opponent reply model, band-conditional: search rolls opponent turns under measured field behavior at our rating band, never under our own priority table | replay mining (D26; per-seat filters) | rollout policy for opponent turns; unlocks the clean 2-ply retest | pending | R3 / post-mining |

## Rules of the table

- Append-only; a superseded entry gets a strikethrough and a successor, never deletion.
- "Gate result" is a number with a game count, or the entry does not ship.
- Negative results stay in the table struck through — they are evidence too
  (see: hysteresis margin 500, −14 pts; 2-ply under fictional opponent, −9 pts;
  ES/TD weight tuning, null vs defaults).
