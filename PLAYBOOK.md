# PLAYBOOK — the agent's strategy, as an auditable table

*Every strategic behavior modifier in the agent is an entry here, or it is not
strategy. Each entry carries three mandatory fields: the **measurement** that
learned it, the **behavior** it drives, and the **gate** it passed before
shipping. Entries a judge can recompute. (Austin's ledger: D23, D25.)*

| # | Entry | Learned from (measurement) | Drives (behavior) | Gate result | Status |
|---|---|---|---|---|---|
| C1 | Energy trajectory: per-Pokemon expected energy at t..t+3, both sides (ours exact from attach rate + outs arithmetic; theirs from posterior + observed attachments) | `outs.py` hypergeometrics (10/10 self-checks); posterior graded 0.63@t1→0.94@t10 | the trajectory features every entry below reads | pending | BUILDING |
| C2 | Damage potential: the threat ladder — max damage each side can present at t..t+3, from C1 + attack costs + retreat constraints; posterior-mixed archetype curves (D13) prior the unseen | D13 goldfish curves with quantile bands; C1 | named trajectory features in `_evaluate`; targets the measured mid-game blindness (58% of positions in one 0.52 bin) | pending | BUILDING |
| C3 | Trajectory verdict: project C2 and map through the calibrated evaluator — P(win) on the current expected path, with the crossover turn | calibration fit (ECE 0.008, 23k held-out samples); C2 | the printed verdict per turn; input to C4 | pending | QUEUED |
| C4 | Risk posture: candidate lines scored mean + λ·spread over determinization outcomes; λ > 0 when C3 verdict is bad (buy variance when losing), λ < 0 when ahead (protect the win) | determinization value spreads (already computed, currently discarded); C3 | replaces mean-only line selection; supersedes the "aggression knob" | pending | QUEUED |
| 4 | Opponent reply model, band-conditional: search rolls opponent turns under measured field behavior at our rating band, never under our own priority table | replay mining (D26; per-seat filters) | rollout policy for opponent turns; unlocks the clean 2-ply retest | pending | R3 / post-mining |
| — | ~~Determinization width scales to budget~~ | 900-game gate: 450-450 at equal bank; override rate 53.9% vs 54.2% — converges by N=3; 1-ply eval is the wall | (reverted; comment-block measurement in main.py, commit 4b729b2) | REJECTED | STRUCK |

## Rules of the table

- Append-only; a superseded entry gets a strikethrough and a successor, never deletion.
- "Gate result" is a number with a game count, or the entry does not ship.
- Negative results stay in the table struck through — they are evidence too
  (see: hysteresis margin 500, −14 pts; 2-ply under fictional opponent, −9 pts;
  ES/TD weight tuning, null vs defaults).
