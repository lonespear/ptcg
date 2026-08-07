# Interaction mining: what the dead forest knows, named and screened

The question this answers: the D34 tree leaf (playbook entry 7) described the
held-out day better than the shipped linear margin (AUC 0.692 against 0.661)
and then lost its gate at 0.2135 pooled. If the description was real, some of
it should be liftable into named terms the linear spine can carry. This report
computes exact SHAP interaction values over the refused forest, hand-lifts the
strongest pairs as named product terms plus hinge terms on single features,
and screens every candidate the way every shipped weight was screened. A third
family screens beside them, Austin's nomination of 2026-08-07: expected
incoming damage, the probabilistic replacement for `_threat_at`'s
deterministic threshold and its copy-growth-to-every-slot allocation. No gate
games were run; this is the nomination stage only.

**The answer is that nothing earned a nomination.** All five product terms and
all eight hinge terms fail the statistical screen on the held-out day, singly
and jointly, and the five expected-damage terms fail it harder: significant in
sample with the wrong sign, and significantly WORSE on held-out logloss. The
forest's held-out edge over the linear description is real but diffuse: the
top five pairs together hold 18% of the pairwise interaction mass, and
entering all eighteen candidates at once moves held-out AUC by -0.005
[-0.013, +0.004]. The negative results are the findings, and they sharpen two
standing diagnoses at once: entry 7's (the forest's advantage is hundreds of
small local corrections, not a handful of nameable interactions) and entry
6's (a threat count on the opponent's board describes positions winners
reach, in expectation form exactly as in threshold form).

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
describes. No candidate reached it: every one failed the first stage on the
held-out day, and a term that does not even describe tomorrow's winners is
not a candidate for steering tomorrow's decisions. The two in-sample-real
mined terms (prize_diff x hp_diff, hinge(prize_diff, -1)) are the classic
screen catch: 7,500 training episodes will certify curvature that a
1,286-episode held-out day cannot detect, and the shipped-weight bar is
held-out transfer, not in-sample z. The expected-damage family is the other
catch: there the first stage already returns the Goodhart verdict, because
the fitted sign says the term describes the positions winners reach rather
than a danger worth paying to avoid.

## Nominations for the gate queue

None, from any of the three families. The recommendation to the phase-weights
agent is to spend the gate budget elsewhere: this analysis bounds the
recoverable-by-named-terms share of the forest's edge at roughly zero (joint
arm -0.001 AUC [-0.007, +0.005] for the mined terms against the forest's own
+0.017 over the same-features linear model), the expected-damage family is
refused with the wrong sign, and the next attempt on the leaf axis should be
something other than product and hinge lifts of pairwise structure, because
the structure is diffuse by measurement, not by suspicion.

## Reproduce

    /usr/bin/python3 scripts/mine_interactions.py shap
    ~/Desktop/PTCG_AI/.venv/bin/python scripts/mine_interactions.py expected
    /usr/bin/python3 scripts/mine_interactions.py screen

Inputs are the committed `tree_leaf.json`, `tree_features.csv` and
`data/mined/*/positions.jsonl.gz`; outputs are `interaction_shap.json`,
`expected_incoming_features.csv` and `interaction_screen.json` beside this
file.
