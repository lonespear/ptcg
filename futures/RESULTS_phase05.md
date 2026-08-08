# Phase 0.5 results — sampler realism with engine legality (holdout 2026-08-06)

Verdict of record, under the same pre-registered criteria as Phase 0
(README.md A-D and the 70% coverage rule, unchanged):
**NO-GO — archetypes passing A-D cover 0.0% of holdout turns.**

Phase 0 blamed its failures on the offline availability surrogate. Phase 0.5
removes that excuse: every menu is the engine's real legal menu, every hidden
zone is dealt from the replay-derived deck priors through the agent's own
posterior (PRIOR_ART.md's determinization requirement), and the failure that
remains is the sampler's. Numbers regenerate with:

    ~/miniforge3/bin/python3 futures/validate_sampler_engine.py   # → results_engine.json (of record)
    ~/miniforge3/bin/python3 futures/diag_engine.py               # → results_engine_diag.json (diagnostic)

(Python ≥ 3.10 required by the engine bindings; /usr/bin/python3 is 3.9.)

## Design

Every distinct holdout seat-turn in the positions sample is a playout root:
1,101 turns (Grimmsnarl 366, Fezandipiti 202, Lopunny 190, Kangaskhan 109,
Dragapult 71, Ogerpon 67; Dudunsparce and Archaludon never appear in the
sample). Each root is entered at the earliest sampled main-menu visit of that
turn (170 at ordinal 0, median entry ordinal 3) and played K=5 times inside
the engine via `search_begin`/`search_step`. The played seat's hand and board
are real (it is that seat's own logged observation); its deck and prizes, and
all three hidden zones of the other seat, are dealt per playout from the
deck-priors posterior, decklists sampled proportional to posterior weight.
`sampler.sample_action()` picks the action type on each real main menu; the
concrete option follows the Phase 1 sub-choice rules the agent ships
(`_opp_order`: cheapest-KO-else-biggest-hit attack, Active attach, first
option otherwise); every other prompt runs the agent's rule policy, exactly
as search rollouts do. Chip and prizes are measured in-engine at the
defender's next main menu against the series.parquet top-of-turn baseline
(verified equal to the observation board on 170/170 ordinal-0 roots;
1,090/1,101 overall, the 11 exceptions being real prefix knockouts that both
sides of the pairing share). The real side is the same construction Phase 0
used, on the same turns: a paired comparison. Action-mix TV uses suffix
visits (ordinal ≥ entry) on both sides; the attack criterion is undiluted
because a real prefix cannot contain an attack. All 5,505 playouts
completed; 2 hit the 25-visit cap.

## Pre-registered test — per archetype

Criteria: A action-mix TV ≤ 0.10; B chip |Δmean| ≤ max(10, 15%) and W1 ≤ 20;
C |Δprizes| ≤ 0.05; D |Δattack rate| ≤ 5 pts. n = real holdout turns played.

| Archetype | n | TV | chip real→samp | W1 | prizes real→samp | attack real→samp | failed | verdict |
|---|---|---|---|---|---|---|---|---|
| Marnie's Grimmsnarl ex | 366 | 0.083 | 47.6→47.6 | 3.3 | 0.458→0.236 | 0.541→0.401 | C, D | NO-GO |
| Fezandipiti ex | 202 | 0.016 | 22.8→21.5 | 4.5 | 0.611→0.329 | 0.658→0.576 | C, D | NO-GO |
| Mega Lopunny ex | 190 | 0.056 | 50.4→37.3 | 13.2 | 0.391→0.241 | 0.632→0.510 | B, C, D | NO-GO |
| Mega Kangaskhan ex | 109 | 0.154 | 27.4→20.8 | 6.8 | 0.394→0.125 | 0.541→0.297 | A, C, D | NO-GO |
| Dragapult ex | 71 | 0.055 | 36.1→27.2 | 10.5 | 0.386→0.160 | 0.690→0.482 | C, D | NO-GO |
| Teal Mask Ogerpon ex | 67 | 0.110 | 29.8→28.1 | 2.7 | 0.561→0.545 | 0.687→0.684 | A (by 0.010) | NO-GO |
| Dudunsparce † | 0 | — | — | — | — | — | no data | NO-GO |
| Archaludon ex † | 0 | — | — | — | — | — | no data | NO-GO |
| FIELD (all) | 1101 | 0.040 | 38.7→34.2 | 4.6 | 0.471→0.257 | 0.606→0.469 | C, D | NO-GO |

† below the tables' TOP_ARCH cut and absent from the holdout positions
sample (0.29% of holdout turns combined). The six played rows cover 90.9% of
holdout turns; GO rows cover 0%. The ordinal-0-only robustness subset agrees
(FIELD n=170: TV 0.053, attack 0.535→0.369, prizes 0.408→0.150).

Flags: suffix attach rate is low everywhere (field 49.1% real vs 39.0%
sampled). Disruption-per-play sits inside the holdout 95% interval for every
archetype except Mega Kangaskhan (field 0.059 real vs 0.064 sampled).

## What moved between Phase 0 and Phase 0.5

Real legality fixed what Phase 0's decomposition said it would fix, and only
that. Criterion A now passes for four of six played archetypes (field TV
0.084 → 0.040); chip means land within 1-2 HP for Grimmsnarl, Fezandipiti and
Ogerpon. The kill relocated to the turn-level observables no decision-level
test sees: sampled turns are short (3.91 suffix visits vs 4.56 real, field),
attack 13.7 points less often, and score far fewer knockouts when they do
attack. Phase 0's menu replay (TV 0.043 on real menus) was accurate and is
unchanged; a policy that matches the field per decision still drifts per
turn, because per-visit errors compound and because its choices steer the
state into menus the field never faces (attack legal on only 28% of sampled
ordinal-0 menus, rising to 43% at 3+).

## Decomposition (post-verdict diagnostics; the gate stands as computed)

1. **Two failure axes, roughly equal weight in the prize gap.** Field prize
   deficit −0.214/turn ≈ attack-rate share (−0.137 × 0.785 ≈ −0.108) plus
   conditional share (0.469 × (0.547 − 0.785) ≈ −0.111).
2. **Attack-rate deficit: late-turn end excess plus early exhaustion.**
   End_turn hazard on multi-option menus: 1.9% vs 2.4% real at ordinal 0,
   2.2% vs 2.8% at 1, then crossing to 4.6% vs 3.0% at 2 and 7.6% vs 5.0% at
   3+. The `NoEarlyEndSampler` variant (end_turn only when it is the sole
   type offered) halves the deficit (field Δattack −6.5 pts) and still fails
   D on four archetypes; coverage stays 0%.
3. **Conditional knockout deficit: the hard core.** Real attacking turns
   take 0.785 prizes (n=643); sampled attacking turns take 0.547 (n=2,480),
   and the no-early-end variant leaves this at 0.524 — forcing more attacks
   makes the average attack worse. The field times its attacks
   endogenously: it attacks when the knockout is on the table, and it builds
   that table first (attach targets, evolutions, gust effects such as Boss's
   Orders with a chosen victim). A per-visit type propensity cannot express
   "attack because this one kills", and the Phase 1 sub-choice rules
   (first-option play, rule-policy card targets) spend gust and setup cards
   without the line they exist for. Chip passing while prizes fail is the
   same fact seen twice: damage that should convert to knockouts stays on
   the board as chip.

## Caveats

The test flatters the sampler where it deviates from the Phase 1 condition:
the played seat's hand, board and prefix are real, and mid-turn entries share
the real prefix in the chip/prize observables. Phase 1 rollouts would
determinize the hand too and compound over 2-3 full turns, so the divergence
measured here is a floor. The play sub-choice rule is part of the machinery
under test; some of the conditional-knockout deficit belongs to it, and the
diagnostics cannot split its share from the type tables' without a better
sub-choice model to compare against.

## Disposition

Phase 0.5 was pre-registered as the genuine kill test, and it killed: the
flat stochastic turn-sampler, given real legality and honest determinization,
prices the opponent's turn at roughly half its real prize output. Rollouts
built on it would score futures against an opponent measurably softer than
the field — the "confidently wrong" failure three external teams measured
Elo losses from (PRIOR_ART.md). Path B stays paused; rollout engineering
does not start on this opponent model. What survives, verified: the
menu-conditioned reply tables at decision level (TV 0.043), the damage/prize
head (chip within 5 HP at field level), the priors/posterior determinization
stack, and this harness — 5,505 engine playouts from mined positions in 10
seconds, ready to re-gate any future opponent model (value-aware attack
timing, learned sub-choices) under the same frozen tolerances.
