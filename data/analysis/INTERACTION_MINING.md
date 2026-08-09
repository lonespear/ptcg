# Interaction mining: what the dead forest knows, named and screened

The question this answers: the D34 tree leaf (playbook entry 7) described the
held-out day better than the shipped linear margin (AUC 0.692 against 0.661)
and then lost its gate at 0.2135 pooled. If the description was real, some of
it should be liftable into named terms the linear spine can carry. This report
computes exact SHAP interaction values over the refused forest, hand-lifts the
strongest pairs as named product terms plus hinge terms on single features,
and screens every candidate the way every shipped weight was screened. Three
further families screen beside them, Austin's nominations of 2026-08-07:
expected incoming damage (the probabilistic replacement for `_threat_at`'s
deterministic threshold and its copy-growth-to-every-slot allocation), the
evolution-aware threat ladder (a coverage repair: the shipped ladder never
sees a benched basic's future evolved attacks), and the discounted threat
integral (a gradient over horizons k = 0..5 where the shipped term is a
threshold at k = 3). No gate games were run; this is the nomination stage
only.

**One candidate earned a nomination, and it is the composition of the two
ladder repairs: the discounted threat integral over evolution-aware
profiles.** At gamma 0.5-0.8 it passes both intervals of the screen —
training coefficient and held-out logloss improvement each clear of zero —
while either repair alone does not. Everything else failed. All five product
terms and all eight hinge terms are null on the held-out day, singly and
jointly; the five expected-damage terms fail harder, significant in sample
with the wrong sign and significantly WORSE on held-out logloss. The forest's
held-out edge over the linear description is real but diffuse: the top five
pairs together hold 18% of the pairwise interaction mass, and entering all
eighteen mined candidates at once moves held-out AUC by -0.005 [-0.013,
+0.004]. The negative results sharpen two standing diagnoses: entry 7's (the
forest's advantage is hundreds of small local corrections, not a handful of
nameable interactions) and entry 6's (a threat count on the opponent's board
describes positions winners reach, in expectation form exactly as in
threshold form).

## Instrument

`scripts/mine_interactions.py shap` extends the repo's exact path-dependent
TreeSHAP (`scripts/shap_traces.py`) with present/absent conditioning: the
pairwise value is Phi[i,j] = (phi_j given x_i present - phi_j given x_i
absent) / 2, computed per tree with training-cover expectations, over the
committed 419-tree forest (`tree_leaf.json`) and the committed sample
(`tree_features.csv`, day split as fitted: train 07-31..08-05, held out
08-06, 7,506 / 1,286 episodes, one position each, earliest turn, rating >=
1000). Verified against brute-force coalition enumeration of the interaction
index on a truncated forest to max |diff| 1.57e-15, and each pair is computed
twice (condition on i, condition on j) with max asymmetry 6.9e-17 over all
1,286 held-out positions. Units are log-odds of winning.

## What the forest's structure looks like

Mean |phi| on the held-out day, main effects, top of the table: prize_diff
0.339, hp_diff 0.205, hand_diff 0.193, threat_traj 0.124, damage_diff 0.097.
The strongest pairwise interaction is an order of magnitude below the
strongest main effect:

| rank | pair | mean abs interaction | mean signed |
|---|---|---|---|
| 1 | prize_diff x hp_diff | 0.0481 | -0.0160 |
| 2 | prize_diff x prizes_left_me | 0.0296 | +0.0106 |
| 3 | threat_traj x energy_traj | 0.0279 | -0.0088 |
| 4 | hp_diff x hand_diff | 0.0241 | +0.0009 |
| 5 | energy_diff x energy_traj | 0.0241 | -0.0111 |
| 6 | energy_diff x hand_diff | 0.0236 | -0.0033 |
| 7 | prize_diff x hand_diff | 0.0225 | -0.0004 |
| 8 | prize_diff x damage_diff | 0.0184 | +0.0015 |
| 9 | prize_diff x bench_diff | 0.0181 | -0.0004 |
| 10 | threat_traj x hand_diff | 0.0168 | -0.0006 |

Aggregate mass: the 120 pairs sum to 0.847 of mean-|Phi| against 1.530 for
the 16 main effects, so pairwise structure is a real part of what the forest
computes; but the mean pair holds 0.007 and the top five together hold 0.154.
Diffuse, in the exact sense the entry-7 diagnosis predicted.

## The screen

`scripts/mine_interactions.py screen`, artifacts in
`interaction_screen.json`. Candidates enter one at a time on top of the terms
the shipped evaluator scores (prize_diff, hp_diff, energy_diff, bench_diff,
damage_diff, threat_traj, no_active_me, went_first, turn fixed effects),
logistic MLE on the six training days; the held-out day then scores the
candidate model against the base model, paired bootstrap, 2,000 resamples.
Passing requires the training coefficient interval AND a held-out improvement
interval (delta AUC up or delta logloss down) to exclude zero. Base model
held-out: AUC 0.6681, logloss 0.6534. Coefficients read per training SD of
the raw term. Hinge knots are training-day quartiles, degenerate knots
dropped.

| candidate | SHAP strength | beta/SD [95% CI] | held-out dAUC [95% CI] | held-out dLL [95% CI] | verdict |
|---|---|---|---|---|---|
| prize_diff*hp_diff | 0.0481 | **-0.121 [-0.190, -0.051]** | -0.0006 [-0.0031, +0.0018] | +0.0006 [-0.0019, +0.0031] | REFUSED: no held-out transfer |
| prize_diff*prizes_left_me | 0.0296 | -0.058 [-0.304, +0.188] | +0.0003 [-0.0001, +0.0007] | -0.0001 [-0.0004, +0.0003] | REFUSED: unidentified (r = 0.95 with prize_diff) |
| threat_traj*energy_traj | 0.0279 | +0.016 [-0.048, +0.080] | +0.0006 [-0.0001, +0.0014] | -0.0002 [-0.0006, +0.0001] | REFUSED: null both ways |
| hp_diff*hand_diff | 0.0241 | -0.003 [-0.059, +0.053] | +0.0001 [-0.0001, +0.0003] | -0.0001 [-0.0001, +0.0000] | REFUSED: null both ways |
| energy_diff*energy_traj | 0.0241 | -0.003 [-0.059, +0.054] | -0.0000 [-0.0002, +0.0001] | +0.0000 [-0.0001, +0.0001] | REFUSED: null both ways |
| hinge(prize_diff, -1) | | **-0.261 [-0.411, -0.111]** | -0.0007 [-0.0036, +0.0024] | +0.0005 [-0.0020, +0.0029] | REFUSED: no held-out transfer |
| hinge(prize_diff, 0) | | -0.059 [-0.163, +0.044] | -0.0000 [-0.0012, +0.0011] | -0.0002 [-0.0010, +0.0007] | REFUSED: null both ways |
| hinge(energy_diff, -1) | | -0.050 [-0.159, +0.058] | -0.0000 [-0.0011, +0.0011] | +0.0000 [-0.0007, +0.0007] | REFUSED: null both ways |
| hinge(energy_diff, 0) | | -0.035 [-0.128, +0.058] | +0.0003 [-0.0006, +0.0013] | -0.0001 [-0.0007, +0.0004] | REFUSED: null both ways |
| hinge(energy_diff, 1) | | -0.047 [-0.129, +0.035] | +0.0004 [-0.0009, +0.0018] | -0.0003 [-0.0012, +0.0006] | REFUSED: null both ways |
| hinge(threat_traj, -10) | | +0.052 [-0.083, +0.187] | +0.0002 [-0.0008, +0.0013] | +0.0002 [-0.0004, +0.0007] | REFUSED: null both ways |
| hinge(threat_traj, 20) | | +0.081 [-0.031, +0.193] | +0.0002 [-0.0018, +0.0020] | +0.0003 [-0.0007, +0.0013] | REFUSED: null both ways |
| hinge(threat_traj, 80) | | +0.083 [-0.004, +0.169] | -0.0003 [-0.0027, +0.0022] | +0.0005 [-0.0009, +0.0018] | REFUSED: null both ways |
| exp_in_t1_max | | **+0.135 [+0.071, +0.199]** | -0.0015 [-0.0063, +0.0031] | **+0.0030 [+0.0001, +0.0059]** | REFUSED: wrong sign, held-out harm |
| exp_in_t2_max | | **+0.134 [+0.071, +0.197]** | -0.0015 [-0.0064, +0.0032] | **+0.0029 [+0.0000, +0.0058]** | REFUSED: wrong sign, held-out harm |
| exp_in_t1_split | | **+0.130 [+0.066, +0.194]** | -0.0023 [-0.0063, +0.0015] | **+0.0031 [+0.0005, +0.0058]** | REFUSED: wrong sign, held-out harm |
| exp_in_t2_split | | **+0.123 [+0.060, +0.186]** | -0.0021 [-0.0059, +0.0015] | **+0.0029 [+0.0004, +0.0055]** | REFUSED: wrong sign, held-out harm |
| exp_net_t2 | | -0.054 [-0.139, +0.032] | -0.0005 [-0.0018, +0.0009] | +0.0006 [-0.0002, +0.0014] | REFUSED: null, point sign wrong |

Joint arms, same base, same bootstrap:

| arm | k | held-out AUC | dAUC [95% CI] | dLL [95% CI] |
|---|---|---|---|---|
| top-5 products together | 5 | 0.6686 | +0.0005 [-0.0026, +0.0036] | +0.0005 [-0.0022, +0.0031] |
| all hinges together | 8 | 0.6661 | -0.0021 [-0.0075, +0.0033] | +0.0008 [-0.0027, +0.0043] |
| expected-damage family | 5 | 0.6647 | -0.0035 [-0.0099, +0.0028] | **+0.0037 [+0.0001, +0.0074]** |
| everything together | 18 | 0.6637 | -0.0045 [-0.0130, +0.0041] | +0.0043 [-0.0010, +0.0097] |

## Expected incoming damage (Austin's nomination, 2026-08-07)

The shipped `_threat_at` has two identified gaps: it is a deterministic
threshold (the hardest attack payable at t+k against the mean growth curve),
and it feeds the full board-level growth to every opponent slot when checking
payability, so three 1-Energy bodies are credited as if every attachment
could land on each of them simultaneously. The family fixes both. Per
opponent Pokemon with attack profile p_j and attached Energy e_j, the
expected hardest payable attack under an attach-count distribution: the count
of Energy arriving by t+k is Poisson with mean read off the SAME archetype
growth curve the shipped projection uses (`_traj_growth`, accel already in
the observed curve; documented in `mine_interactions.py::_poisson_pmf`), and
the growth goes to ONE attacker at a time — the max variant gives the whole
budget to whichever slot it makes most dangerous, the split variant thins the
Poisson evenly across the attackers. Never copied to all slots. For a
one-attack Pokemon the term is exactly P(online by t+k) x damage. Features:
exp_in_t1/t2 in both allocations (their board), and exp_net_t2, our
expected-outgoing max minus theirs. Computed by the agent's own machinery
over the same 8,792 episodes (`expected_incoming_features.csv`; 0.08 ms a
position). The distributions are healthy: median expected incoming 35-50
damage, 4-5% zeros, max and split correlated 0.90.

The precedent question the screen was to answer: `online_lead`, the
threshold form of turns-to-online, measured null (C1) — does the expectation
form price where the threshold form did not? **It does not price; it
anti-prices.** All four incoming terms fit positive — MORE expected incoming
damage predicts WINNING — at +0.12 to +0.14 per SD with intervals clear of
zero, on top of a base model that already holds threat_traj, hp_diff and
energy_diff fixed. That is not a subtle effect of allocation (max and split
agree) or of horizon (t+1 and t+2 agree), and it is mid-and-late-game
(terciles: null on turns <= 3, +0.17-0.22 significant on 4-6, +0.08-0.11 past
6). On the held-out day every incoming variant makes prediction WORSE by a
logloss interval that excludes zero (+0.0029 to +0.0031), the only candidates
in this report to manage that. The net differential exp_net_t2 is null with
the point estimate also wrong-signed (-0.054 [-0.139, +0.032]: our expected
firepower exceeding theirs reads as losing).

The reading is entry 6 verbatim, in expectation form: an opponent board that
threatens damage soon is a developed board, and developed opponent boards are
what the games winners are winning look like once the differentials are held
fixed — a pace description, not a danger price. The decision-value test was
never reached, and for this family the first stage already answers it in the
Goodhart direction: a weight on any of these terms would PAY the search to
walk toward positions where the opponent threatens more, or (as a penalty,
sign flipped by hand against the fit) would price a description the data
says runs the other way. Both defects Austin named in `_threat_at` are real
as code critique; fixing them does not produce a feature that prices. The
family is refused with a finding, and the finding is that the inversion the
exposure count showed at threshold form survives the upgrade to expectation
form intact.

## The threat ladder repairs (Austin's nominations, 2026-08-07)

Two defects in the shipped C2 term, verified in code, one repair each, and
the repairs compose.

**The coverage defect.** `_attack_profile(card_id)` reads only the card's own
printed attacks, so `_threat_at`'s ladder never sees a benched basic's future
evolved attacks — a Charmander with three Energy contributes Charmander's
attacks, never Charizard's, on both sides of the board. The EVOLUTION-AWARE
LADDER unions each slot's profile with the attacks of evolutions reachable
within the horizon: one evolution step per own turn (the slots are already in
play, so s steps need s of the next k turns), attached Energy surviving
evolution, and the evolution cards priced by outs arithmetic — a copy in hand
is certain, an unseen copy prices hypergeometrically (`ptcg.creation.outs`)
with the draws its step schedule allows, chain steps multiplying under a
stated independence approximation. Pools are the posterior top-1 list for
their side and, at fit time, the same posterior pointed at our own board for
ours (the runtime agent would use its exact list; the mined focal seat's list
is unknown). A clock-only variant prices nothing and counts any chain the
pool contains: the pure-coverage envelope. Everything else about `_threat_at`
— full board growth to every slot, hardest payable attack, max over slots —
is kept, so the comparison isolates coverage.

**The shape defect.** The shipped term is a threshold at one horizon: a stoke
that closes distance without crossing t+3 payability earns nothing, online at
t+1 scores the same as online at t+3, and past k = 3 the placement-blind 30
per Energy is all that remains. The DISCOUNTED THREAT INTEGRAL is the
gradient form: sum over k = 0..5 of gamma^k times the threat-at-k
differential, on shipped and on evolution-aware profiles, gamma profiled over
0.4-0.8 rather than optimized. The single-horizon prior (the C2 fit chose
k = 3 over k = 4/5) says the far tail is noisy; the discount is how tail
signal survives without tail noise, and a profile collapsing toward the k = 3
spike would mean the point term was already right.

**Incidence — the repairs are not rare situations.** The recomputed shipped
ladder matches the committed tree_features column to 0.0000 drift (the join
and the reimplementation are both validated). The weighted evolution repair
changes the k = 3 ladder on **31.7%** of positions (16.5% our side, 18.9%
theirs; mean 50.5 damage when changed; stable train 31.6% / held-out 32.2%),
the clock-only envelope on 45.9% (mean 106.8 damage). The per-k profile the
integral sums is non-flat on 78.2% (shipped) and 84.4% (evolution-aware) of
positions, with a mean spread above 100 damage. A null here would have been a
mispricing verdict, not a rarity verdict. It is not needed:

| arm | beta/SD [95% CI] | held-out dAUC [95% CI] | held-out dLL [95% CI] |
|---|---|---|---|
| swap evo point k=3 (weighted) | **+0.115 [+0.064, +0.167]** | +0.0014 [-0.0013, +0.0040] | -0.0005 [-0.0017, +0.0007] |
| both: shipped / evo point | -0.065 [-0.185, +0.056] / **+0.171 [+0.055, +0.288]** | +0.0019 [-0.0021, +0.0056] | -0.0007 [-0.0024, +0.0010] |
| swap evo point k=3 (clock-only) | **+0.072 [+0.022, +0.123]** | +0.0021 [-0.0011, +0.0054] | -0.0004 [-0.0019, +0.0011] |
| both: shipped / clock-only | **+0.082 [+0.004, +0.160]** / +0.016 [-0.058, +0.089] | +0.0004 [-0.0003, +0.0011] | -0.0001 [-0.0004, +0.0002] |
| integral, shipped, g=0.5 | **+0.066 [+0.007, +0.124]** | +0.0017 [-0.0009, +0.0045] | -0.0009 [-0.0022, +0.0004] |
| integral, shipped, g=0.8 | **+0.076 [+0.020, +0.133]** | +0.0013 [-0.0005, +0.0031] | -0.0006 [-0.0014, +0.0002] |
| integral, evo, g=0.4 | **+0.075 [+0.017, +0.134]** | +0.0024 [-0.0006, +0.0052] | -0.0014 [-0.0028, +0.0001] |
| **integral, evo, g=0.5** | **+0.084 [+0.026, +0.141]** | +0.0025 [-0.0002, +0.0051] | **-0.0014 [-0.0026, -0.0001]** |
| **integral, evo, g=0.6** | **+0.091 [+0.035, +0.147]** | +0.0024 [-0.0000, +0.0049] | **-0.0013 [-0.0026, -0.0002]** |
| **integral, evo, g=0.7** | **+0.097 [+0.041, +0.152]** | +0.0023 [+0.0000, +0.0047] | **-0.0013 [-0.0025, -0.0001]** |
| **integral, evo, g=0.8** | **+0.100 [+0.046, +0.155]** | **+0.0022 [+0.0000, +0.0046]** | **-0.0012 [-0.0024, -0.0001]** |

(All arms replace threat_traj in the base model, so every delta is the
"variant replaces the shipped point term" comparison. Full profile in
`interaction_screen.json` under `evolution_ladder`.)

**Readings.**

- **The evolution-aware point term subsumes the shipped one, and the
  availability arithmetic is what does it.** In the side-by-side arm the
  weighted evo term takes the whole load (+0.171, interval clear) and the
  shipped term goes null (-0.065 [-0.185, +0.056]) — the C2 pattern again,
  where the better-formed version of a feature kills its predecessor in the
  joint fit. The clock-only envelope does NOT manage this (shipped stays
  significant, the envelope goes null): counting a Charizard that is three
  hypergeometric misses away as certain damage adds noise, pricing it by
  outs arithmetic adds signal.
- **Alone, each repair fails the transfer interval.** The evo point term's
  held-out deltas are the right direction and null; so are all five shipped
  integrals. In-sample, every one of these betas is clear of zero — the
  screen's whole function is that this does not suffice.
- **Together they pass.** The evolution-aware integral at gamma 0.5-0.8
  clears both intervals: training z of 2.9-3.6 and held-out logloss
  improvement of -0.0012 to -0.0014 with the 95% interval excluding zero
  (at g = 0.7/0.8 the AUC interval touches zero from above as well). The
  gamma profile is a plateau over 0.5-0.8, not a spike near the k = 3
  weighting — with evolutions visible, the tail horizons carry signal the
  point term discards, which is the shape Austin predicted when nominating
  the discount as the way tail signal survives tail noise.
- **Honesty about size and multiplicity.** The effect is a few thousandths:
  +0.002 AUC, -0.0013 logloss, against the forest's +0.024 held-out edge
  over this base. Fourteen ladder arms were screened and four passed; they
  are one family under a smooth parameter, all passing in the same
  direction, which is the pattern of one real effect rather than of
  fourteen chances — but a reader should know the count, and the gate, not
  this screen, is the arbiter.

## Readings, per candidate worth a sentence

- **prize_diff x hp_diff** (the forest's strongest pair, and the strongest
  in-sample candidate). Both instruments agree on the sign: the SHAP signed
  interaction is -0.016 and the fitted product coefficient is -0.121 per SD
  with an interval clear of zero. The reading is redundancy of leads: a prize
  lead and an HP lead together are worth less than the sum of their parts.
  But the tercile fit says it is not one effect: +0.064 on turns <= 3, -0.028
  on 4-6, -0.154 (significant) past 6. That is turn-dependence of lead value
  wearing an interaction's clothes, and it transfers nothing to the held-out
  day: the bootstrap bounds any true AUC gain below +0.002. As a decision
  weight it would tell the search to discount dealing damage while ahead on
  prizes, a behavior nothing here justifies paying for.
- **hinge(prize_diff, -1)** is the same shape read on one axis: concavity in
  the prize count. In-sample, falling behind by a prize costs more per prize
  than a lead pays (-0.261 per SD, interval clear of zero), and again the
  early tercile flips sign and the held-out day is unmoved. Real curvature in
  the training description; no measured decision value.
- **prize_diff x prizes_left_me** is the forest reading "a lead late in the
  race" (endgame proximity). Named as a product it is collinear with
  prize_diff itself (r = 0.95) and the coefficient is unidentified. If this
  structure is ever wanted, it is a threshold on prizes_left_me, and the
  prize hinge row above says that family is not transferring either.
- **The hand_diff rows are the quiet finding.** hand_diff is the forest's
  third-largest main effect (0.193) and the struck row the entry-7 diagnosis
  blamed for the gate collapse; the task rule was extra skepticism before
  letting it back in inside an interaction. No skepticism was needed: its
  interactions with hp_diff, energy_diff, prize_diff and threat_traj are
  nulls at the fourth decimal, in sample and out. The forest's hand_diff
  structure is main-effect Goodhart bait all the way down; there is no
  conditional version of it here that describes winning, let alone one that
  earns a weight.
- **The projection pairs** (threat_traj x energy_traj, energy_diff x
  energy_traj) are the C1/C2 features agreeing with each other; both null.
  The C2 term as shipped is carrying what that family has to give.

## The decision-value test, and why it was not reached

The screen's second stage, the one hand_diff failed at 0.360 over 494 games,
asks whether spending the resource produces the advantage the coefficient
describes. Among the mined and expected-damage families no candidate reached
it: every one failed the first stage on the held-out day, and a term that
does not even describe tomorrow's winners is not a candidate for steering
tomorrow's decisions. The two in-sample-real mined terms (prize_diff x
hp_diff, hinge(prize_diff, -1)) are the classic screen catch: 7,500 training
episodes will certify curvature that a 1,286-episode held-out day cannot
detect, and the shipped-weight bar is held-out transfer, not in-sample z.
The expected-damage family is the other catch: there the first stage already
returns the Goodhart verdict, because the fitted sign says the term
describes the positions winners reach rather than a danger worth paying to
avoid. The one candidate that did reach the test — the evolution-aware
discounted integral — takes its decision-value case to the gate in the
nomination below.

## Nominations for the gate queue

**One: `threat_integral_evo` at gamma 0.6** (the middle of the passing
plateau, and the best training z at 3.16) — the discounted sum over k = 0..5
of the evolution-aware threat differential, replacing the shipped
threat_traj. It is the only candidate of twenty-three screened today to pass
both intervals, and it is the composition of two repairs that each fail
alone, which is itself evidence the effect is the repairs and not the draw.
The decision-value case it takes to the gate: the term pays a gradient for
stoking Energy toward attacks not yet payable and for setup lines whose
evolutions are drawable on schedule — exactly the long-run stoke Austin
named, which the threshold form pays only after the fact. The resources it
prices (attachments, evolution timing) are the ones the search actually
spends, and its inputs resist manufacture: the search cannot draw cards to
inflate availability, and benching a stoke target to raise the integral IS
the setup line the repair exists to see. The Goodhart risk to watch at the
gate is the our-side pool inference: a runtime agent knows its own list
exactly, so the gate build should read `agent/deck.csv` for our side rather
than the posterior stand-in this fit was forced to use. Expected effect size
is small (+0.002 AUC, -0.0013 logloss held out), so the gate arm should be
sized like a C2 gate, not like a rescue.

Nothing else is nominated. The mined-interaction and hinge families bound
the recoverable-by-named-terms share of the forest's edge at roughly zero
(joint arm -0.001 AUC [-0.007, +0.005] against the forest's own +0.017 over
the same-features linear model); the expected-damage family is refused with
the wrong sign; the evolution point term and the shipped-profile integral
are each right-directioned and null alone, and their measured content is
inside the nominated composition.

## Reproduce

    /usr/bin/python3 scripts/mine_interactions.py shap
    ~/Desktop/PTCG_AI/.venv/bin/python scripts/mine_interactions.py expected
    ~/Desktop/PTCG_AI/.venv/bin/python scripts/mine_interactions.py evo
    /usr/bin/python3 scripts/mine_interactions.py screen

Inputs are the committed `tree_leaf.json`, `tree_features.csv` and
`data/mined/*/positions.jsonl.gz`; outputs are `interaction_shap.json`,
`expected_incoming_features.csv`, `evo_threat_features.csv` and
`interaction_screen.json` beside this file.
