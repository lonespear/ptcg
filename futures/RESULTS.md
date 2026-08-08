# Phase 0 results — sampler realism (holdout 2026-08-06)

Verdict of record, under the pre-registered criteria in README.md:
**NO-GO — archetypes passing A-D cover 33.1% of holdout turns (threshold 70%).**

The diagnostics below were run after that verdict and are labeled as such.
They decompose the failure: the reply tables themselves pass the one test that
isolates them (decision-level replay on real menus), while the turn-level
failures trace to the offline availability surrogate — the stand-in for engine
legality that Phase 0 needs and Phase 1 discards — plus a genuine coverage gap
for the two requested archetypes below the tables' TOP_ARCH cut. Numbers
regenerate with:

    /usr/bin/python3 futures/fit_policy.py
    /usr/bin/python3 futures/validate_sampler.py          # → results.json (v1, of record)
    /usr/bin/python3 futures/validate_sampler.py --persistent   # → results_v2.json (diagnostic)
    /usr/bin/python3 futures/diag_damage_head.py          # → results_head_diag.json (diagnostic)

## Pre-registered test (v1, iid availability) — per archetype

Real n = holdout seat-turns; K=3 sampled per real turn, matched (band, arch,
turn) conditioning. Criteria: A action-mix TV ≤ 0.10; B chip |Δmean| ≤
max(10, 15%) and W1 ≤ 20; C |Δprizes| ≤ 0.05; D |Δattack rate| ≤ 5 pts.

| Archetype | n | TV | chip real→samp | W1 | prizes real→samp | attack real→samp | failed | verdict |
|---|---|---|---|---|---|---|---|---|
| Marnie's Grimmsnarl ex | 20000* | 0.085 | 66.3→71.4 | 5.0 | 0.532→0.581 | 0.619→0.690 | D | NO-GO |
| Mega Lopunny ex | 8195 | 0.089 | 55.1→60.1 | 5.4 | 0.527→0.545 | 0.730→0.744 | — | GO |
| Mega Kangaskhan ex | 9442 | 0.072 | 33.2→31.3 | 2.7 | 0.308→0.338 | 0.505→0.486 | — | GO |
| Fezandipiti ex | 13178 | 0.096 | 21.0→26.1 | 5.1 | 0.500→0.594 | 0.602→0.700 | C, D | NO-GO |
| Dragapult ex | 3976 | 0.079 | 51.2→52.9 | 2.6 | 0.489→0.483 | 0.769→0.741 | — | GO |
| Teal Mask Ogerpon ex | 3633 | 0.102 | 24.0→27.6 | 3.6 | 0.647→0.599 | 0.758→0.713 | A (by 0.002) | NO-GO |
| Dudunsparce † | 150 | 0.167 | 22.2→32.1 | 11.3 | 0.559→0.555 | 0.793→0.664 | A, D | NO-GO |
| Archaludon ex † | 39 | 0.210 | 52.9→31.8 | 23.1 | 0.600→0.752 | 0.872→0.675 | A, B, C, D | NO-GO |
| FIELD (all) | 20000* | 0.084 | 44.8→48.0 | 3.2 | 0.496→0.541 | 0.635→0.672 | — | GO |

\* capped sample of a larger population. † OTHER-backoff: below the tables'
TOP_ARCH cut, sampled from the pooled OTHER cells. The eight rows cover 91.1%
of holdout turns; GO rows cover 33.1%.

Flags (non-gating): energy-attach rate is low everywhere (field 66.1% real vs
47.0% sampled — see divergence 1). Disruption-per-play sits inside the holdout
95% interval for every archetype with data except Mega Lopunny (head 0.018 vs
real 0.049 [0.019, 0.119], n=82). Attack-choice agreement on holdout
multi-attack menus: 0.50 with n=16 — below the n≥30 gate threshold,
uninformative either way.

## Diagnostics (post-verdict)

**1. Menu replay — the tables on real menus, no availability model.** For each
of the 1,073 holdout main menus in the positions sample, the policy's
distribution over the types actually offered vs the field's pick:

| Archetype | n menus | expected-vs-realized mix TV | top-1 | log-loss (uniform) |
|---|---|---|---|---|
| FIELD | 1073 | **0.043** | 0.610 | 0.852 (1.302) |
| Marnie's Grimmsnarl ex | 346 | 0.087 | 0.647 | 0.767 (1.281) |
| Fezandipiti ex | 198 | 0.094 | 0.550 | 0.874 (1.315) |
| Mega Lopunny ex | 192 | 0.091 | 0.620 | 0.935 (1.285) |
| Mega Kangaskhan ex | 105 | 0.127 | 0.505 | 0.949 (1.254) |

Attach expected 13.0% vs realized 13.1% of decisions; attack 11.0% vs 9.2%.
Given real legality, the tables reproduce the field's action mix to within ~4
points TV pooled and beat a uniform policy by 35% in log-loss. This is the
component Phase 1 actually consumes — the engine supplies real legality there.

**2. Availability-surrogate bracket (v2, persistent unlock).** Re-running the
turn-level test with the opposite availability idealization (monotone unlock
instead of iid draws) flips the sign of the attack-rate error everywhere:
Grimmsnarl 61.9% real, 69.0% iid, 53.8% persistent; field 63.5% / 67.2% /
53.0%. Neither surrogate can recover P(type ever available in the turn) from
per-menu marginal availability rates; the truth sits between the two brackets.
v2 verdict: 0% coverage — worse, and equally an artifact.

**3. Damage/prize head in isolation** (drawn against each real turn's real
attacked flag): chip means within 0.4-4.5 HP and W1 ≤ 4.5 for all six table
archetypes (field 44.5→45.5, W1 1.0); prizes within 0.035. Fezandipiti's C
failure above (+0.094 prizes) disappears here (+0.005) — it was the
attack-rate overshoot wearing the head's clothes, not the head.

## The three worst divergences

1. **Energy-attach deficit, availability-induced.** Field 66.1% of real turns
   attach; v1 samples 47.0%, v2 50.0%. Worst: Mega Lopunny 81.7% real vs
   57.0% sampled. The corpus only records availability per *menu* (attach
   ord-0 rate 0.60-0.85, declining because players attach and it leaves the
   menu), so no offline surrogate reproduces "attach becomes available
   mid-turn after a draw." Menu replay shows the choice tables are not at
   fault: attach expected 13.0% vs realized 13.1% at the decision level.
2. **Attack-rate bracket, surrogate-dependent sign.** Grimmsnarl +7.1 pts
   (iid) vs −8.1 (persistent); Fezandipiti +9.8 vs −5.5. Downstream it
   inflated v1 prizes (Fez +0.094) — gone in the head-only draw. Both gate
   failures on the two biggest decks trace here.
3. **OTHER-backoff gap on two of the five requested decks.** Dudunsparce
   (8.2k decisions over seven days) and Archaludon ex (1.6k) fall below the
   TOP_ARCH cut: TV 0.167 / 0.210, attack rate −12.9 / −19.7 pts, and the
   pooled damage head misprices Dudunsparce chip +15.5 HP (W1 17). This is a
   real tables gap, not an artifact — but these decks are 0.29% of holdout
   turns combined.

Example, Grimmsnarl mid-game (turn 5-7, 900-1050 band) — real turns:
`play play attach play play evolve ability evolve ability play` (10 visits),
`ability play play evolve ability attack`; sampled:
`play play attach play evolve play attack`,
`ability evolve play ability evolve play attack`. Qualitatively the same
grammar; the quantitative gaps are the two availability effects above.

## Recommendation

The pre-registered gate failed, so Phase 0 does not certify rollout
engineering — but what failed is the offline stand-in for engine legality,
not the reply tables: on real menus the tables reproduce the field's mix to
0.043 TV, and the damage head prices turns to within ~1 HP at field level.
Proposed next step (Phase 0.5, cheap): wire `sampler.sample_action()` into
the engine's real legal-menu loop on determinized states — the exact
integration Phase 1 needs anyway, ~a day — and re-apply these same
pre-registered tolerances with true availability. GO there unlocks rollout
engineering; NO-GO there is a genuine kill. Separately, if Dudunsparce or
Archaludon matter for the gate decks, they need their own table rows once
volume allows (both are <0.3% of the field today).
