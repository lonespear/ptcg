# PLAYBOOK — the agent's strategy, as an auditable table

*Every strategic behavior modifier in the agent is an entry here, or it is not
strategy. Each entry carries three mandatory fields: the **measurement** that
learned it, the **behavior** it drives, and the **gate** it passed before
shipping. Entries a judge can recompute. (Austin's ledger: D23, D25.)*

| # | Entry | Learned from (measurement) | Drives (behavior) | Gate result | Status |
|---|---|---|---|---|---|
| V | Inferred position value: the evaluator's weights are fitted, not chosen — prize 1000 (anchor), HP 2.6, Energy 30, bench width 153, damage already dealt 4.2, empty Active −4000 | logistic fit on the eventual win over 2,823 mined positions from 14,172 rated ladder games, rescaled to evaluator units (`ptcg/advantage.py`, `data/analysis/REPORT.md` §A2) | every term of `_evaluate`, and through it every search value | **0.580** over 802 decided games vs `8bef47a` (Marnie mirror 0.566/498, Garchomp mirror 0.602/304), posture on; 0.565 posture off; own Ogerpon deck 0.510/304 | ACTIVE |
| C3 | Calibrated verdict: the margin → P(win) table ships in the bundle and the agent reads its own verdict at run time (`_pwin`) | isotonic fit refitted for this weight vector, 2,997 self-play games: **ECE 0.009, Brier 0.233 vs 0.250 base rate**, 24,330 held-out samples, 29 breakpoints (`data/calibration_v2.json`) | the deficit test C4 is keyed on; the per-turn verdict a judge can recompute | shipped inside the V gate above; falsification gate asserts the bundled table loads and `_pwin` is monotone (`scripts/falsify_bundle.py`) | ACTIVE |
| C4 | Comeback posture: below P(win) 0.45 the main menu reorders — attack and attach +1.5, play/ability/retreat −1.25 (`BEHIND_CLOSE_BONUS` / `BEHIND_CHURN_PENALTY`) | comeback regression over 3,190 (t5) and 3,372 (t7) behind seats, every behaviour entered at once on the position, deficit and window: attack +1.41, attach +0.80, play −0.59, ability −0.51, retreat −0.61 (`data/analysis/comeback_B.json`) | replaces the flat priority table whenever the calibrated verdict says we are behind; fires on 9.7% of main-menu picks | **0.580** with it (802 games) against 0.565 without (798 games, same decks and seeds) — carried by the Garchomp arm, 0.602 vs 0.549 | ACTIVE |
| C1 | Energy trajectory: per-Pokemon expected energy at t..t+3, both sides | `outs.py` hypergeometrics (10/10 self-checks); posterior graded 0.63@t1→0.94@t10 | the trajectory features C2 would read | none — the advantage inference could not price a trajectory term, because the mined decision stream carries attachment *tempo* and not Energy in play (attachments vs Energy on board, r = 0.14 at t3 and negative after) | DEFERRED, pending position-level energy reconstruction |
| C2 | Damage potential: the threat ladder at t..t+3 from C1 + attack costs + retreat constraints | D13 goldfish curves with quantile bands; C1 | named trajectory features in `_evaluate` | none — it is downstream of C1 and D30 forbids hand-set stand-ins for what the data could not support | DEFERRED, pending C1 |
| 4 | Opponent reply model, band-conditional: search rolls opponent turns under measured field behavior at our rating band, never under our own priority table | replay mining (D26; per-seat filters) | rollout policy for opponent turns; unlocks the clean 2-ply retest | pending | R3 / post-mining |
| — | ~~Hand differential in the evaluator, at the fitted 97 per card~~ | the fit is unambiguous (hand_diff +97 [40, 155]; the own-hand coefficient turns over to −44 once it is in the model) and the play is still wrong: **0.360 over 494 decided games**, and 0.396 for the whole richer vector carrying it, against 0.539 for the same build without it | (removed; the ablation lives in the WEIGHTS comment block in main.py) | REJECTED | STRUCK |
| — | ~~Own-hand term, +5 per card~~ | dropped with the differential above: same mistake, smaller magnitude, and the fit says its sign is wrong | (removed) | superseded | STRUCK |
| — | ~~Determinization width scales to budget~~ | 900-game gate: 450-450 at equal bank; override rate 53.9% vs 54.2% — converges by N=3; 1-ply eval is the wall | (reverted; comment-block measurement in main.py, commit 4b729b2) | REJECTED | STRUCK |

## Rules of the table

- Append-only; a superseded entry gets a strikethrough and a successor, never deletion.
- "Gate result" is a number with a game count, or the entry does not ship.
- Negative results stay in the table struck through — they are evidence too
  (see: hysteresis margin 500, −14 pts; 2-ply under fictional opponent, −9 pts;
  ES/TD weight tuning, null vs defaults).
- A fitted coefficient is a description of positions winners reach. It earns a
  decision weight only if spending the resource produces the advantage it
  measures — which is what the struck hand-differential row cost us to learn.
