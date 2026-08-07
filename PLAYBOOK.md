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
| C1 | Energy trajectory: where each side's Energy is k of its own turns from now — the archetype-conditional curve for both sides, plus, for us, the accelerator we can see in our own hand and on our own board | curves from `ptcg/trajectory.py` over 193,479 (episode, seat, turn) series rows from 14,182 rated games — held-out bias at k=2 is **−0.058** Energy against **+0.926** for the playbook's one-attach-a-turn line; accelerator rates from `ptcg/energy_mechanics.py` (69 acceleration prints); opponent archetype off `_deck_posterior`, which named the replay's own archetype on **93.6%** of the fitted positions against 15% at turn 1 for a board-derived label | the projection C2 reads; the two features C1 defines itself were both priced and both refused | fitted on 8,792 positions, one an episode, rating ≥ 1000, turn fixed effects: `online_lead` (their earliest attack turn minus ours) **31 [−16, +78]** in the joint fit, **40 [−7, +86]** alone; `energy_traj_t2` (projected Energy differential at t+2) **−27 [−58, +3]**, correlated 0.68 with the Energy term already in the vector. Neither interval excludes zero | MEASURED-NULL — the projection ships because C2 reads it; the two features carry no weight, and stay computable in `_trajectory_terms` so new data can re-price them |
| C2 | Damage potential: the hardest attack each side can pay for two of its own turns from now, ours minus theirs (`threat_traj_t2`) | same logistic on the same 8,792 positions, entered on top of exactly the terms this evaluator scores: **1.42 [0.47, 2.36]**, z 2.93, p 0.0034, prize anchored at 1000 (`scripts/fit_trajectory_features.py`, `data/analysis/trajectory_fit.json`). The projection is what carries it, not the board: the same differential taken on the present board fits 1.25 [0.15, 2.34] alone and goes null at 0.48 [−0.82, +1.79] once the t+2 version sits beside it, while t+2 survives at 1.18 [0.04, 2.32] | a named `WEIGHTS` term inside `_margin`, so search value and the C3 verdict both read it; 0.009 ms a rules-path decision | **0.511** over 1,576 decided games vs `ef84786`, mirror decks, seats swapped, two independent seed blocks (Marnie 0.494/976, Garchomp 0.538/600) — inside the 48–52% band, so it ships on the fitted interval rather than on the mirror, which is the rule for a term whose coefficient excludes zero decisively. Invariant clean (4,381 positions, 0 mismatches); `falsify_bundle.py` asserts both tables load from the bundle and the term moved 15,162 of 15,450 margins | ACTIVE |
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
- Two things about C2 a reader should know before trusting its gate. The
  margin it adds is worth 39 on average and 126 at one standard deviation over
  the 14,000 mined positions, against a C3 table fitted for the vector without
  it; that staleness changes the behind/ahead verdict on **0.91%** of those
  positions, which is why the table was not refitted. And the term is inert in
  an exact mirror of our own Ogerpon list — both sides afford the same best
  attack, so the differential is 0 on 98% of positions — while against the
  field it fires on 98–100%. The mirror gate therefore measures it under the
  one matchup where it has least to say.
